"""Ảnh minh hoạ: box đúng (xanh lá) và box dự đoán (đỏ = sai, vàng = khớp), đường loss.

python -m day10.viz --pred work/e2.jsonl --name E2 --out report/figures
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from day10.common import read_split
from scripts.score import _gt_boxes, _match, parse_output

COLORS = {"gt": (40, 200, 80), "tp": (255, 210, 0), "fp": (230, 40, 40)}


def per_image(entries, preds):
    out = []
    for e in entries:
        parsed = parse_output(preds.get(e["image_id"]))
        pred = parsed or []
        per, _ = _match(_gt_boxes(e), pred)
        tp = sum(v[0] for v in per.values())
        fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        out.append({"entry": e, "pred": pred, "valid": parsed is not None, "tp": tp, "fp": fp, "fn": fn})
    return out


def _iou(a, b):
    from scripts.score import iou_xyxy
    return iou_xyxy(a, b)


def draw(data_dir, item, title, path):
    e = item["entry"]
    im = Image.open(Path(data_dir) / e["image_path"]).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)

    def px(b):
        return (b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h)

    gts = _gt_boxes(e)
    for g in gts:
        d.rectangle(px(g["bbox"]), outline=COLORS["gt"], width=3)
        d.text((px(g["bbox"])[0] + 2, px(g["bbox"])[3] + 1), g["label"], fill=COLORS["gt"])
    for p in item["pred"]:
        hit = any(g["label"] == p["label"] and _iou(g["bbox"], p["bbox"]) >= 0.5 for g in gts)
        color = COLORS["tp"] if hit else COLORS["fp"]
        d.rectangle(px(p["bbox"]), outline=color, width=2)
        d.text((px(p["bbox"])[0] + 2, px(p["bbox"])[1] - 11), p["label"], fill=color)
    band = Image.new("RGB", (w, 22), (20, 20, 20))
    ImageDraw.Draw(band).text((6, 5), title, fill=(255, 255, 255))
    canvas = Image.new("RGB", (w, h + 22))
    canvas.paste(band, (0, 0))
    canvas.paste(im, (0, 22))
    canvas.save(path, quality=88)


def pick_examples(items, n_good=3, n_bad=3):
    def f1(i):
        return 2 * i["tp"] / (2 * i["tp"] + i["fp"] + i["fn"]) if i["tp"] + i["fp"] + i["fn"] else 1.0
    with_boxes = [i for i in items if i["entry"]["objects"]]
    good = sorted(with_boxes, key=lambda i: (-f1(i), -i["tp"]))[:n_good]
    bad_fp = sorted(items, key=lambda i: -i["fp"])[:2]
    bad_fn = sorted([i for i in items if i not in bad_fp], key=lambda i: -i["fn"])[:n_bad - 1]
    return [("good", i) for i in good] + [("bad", i) for i in bad_fp + bad_fn]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data", default="data")
    a = ap.parse_args(argv)
    preds = {}
    for line in open(a.pred, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            preds[r["image_id"]] = r["raw"]
    items = per_image(read_split(a.data, "test"), preds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (kind, item) in enumerate(pick_examples(items), 1):
        e = item["entry"]
        title = (f"{a.name} {e['image_id']}  TP={item['tp']} FP={item['fp']} FN={item['fn']}  "
                 f"(xanh=GT, vang=khop, do=sai)")
        path = out / f"{a.name.lower()}_{k}_{kind}.jpg"
        draw(a.data, item, title, path)
        manifest.append({"file": path.name, "image_id": e["image_id"], "kind": kind,
                         "tp": item["tp"], "fp": item["fp"], "fn": item["fn"],
                         "gt_labels": [o["label"] for o in e["objects"]],
                         "pred_labels": [p["label"] for p in item["pred"]]})
    (out / f"{a.name.lower()}_examples.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
