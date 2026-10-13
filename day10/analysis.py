"""Số liệu phụ cho REPORT.md (không thay thế report/evidence/*.json của score.py):
- bootstrap CẶP (cùng ảnh được rút cho hai setup) cho hiệu F1,
- lỗi của E0, kích thước box theo lớp, thang toạ độ sai, tín hiệu reward hacking của GRPO.

python -m day10.analysis --out report/extra/analysis.json
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from day10.common import read_split
from scripts.labcommon import CLASSES
from scripts.score import _gt_boxes, _match, _prf, _to_norm1000, evaluate, parse_output


def load_preds(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["image_id"]] = r["raw"]
    return out


def per_image_counts(entries, preds):
    rows = []
    for e in entries:
        pred = parse_output(preds.get(e["image_id"])) or []
        per, _ = _match(_gt_boxes(e), pred)
        rows.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return rows


def paired_bootstrap(rows_a, rows_b, n_boot=2000, seed=0):
    """Hiệu F1(b) - F1(a) khi rút lại CÙNG tập ảnh cho cả hai setup."""
    rng, n, diffs = random.Random(seed), len(rows_a), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        fa = _prf(*(sum(rows_a[i][k] for i in idx) for k in range(3)))["f1"]
        fb = _prf(*(sum(rows_b[i][k] for i in idx) for k in range(3)))["f1"]
        diffs.append(fb - fa)
    diffs.sort()
    return {"mean_diff": sum(diffs) / n_boot, "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "p_b_not_better": sum(d <= 0 for d in diffs) / n_boot}


def e0_failures(entries, preds):
    stats = Counter()
    examples = {}
    areas = []
    for e in entries:
        raw = preds[e["image_id"]]
        parsed = parse_output(raw)
        if "```" in raw:
            stats["markdown_fence"] += 1
        if parsed is None:
            stripped = raw.strip()
            if not stripped.endswith("]") and not stripped.endswith("```"):
                key = "invalid_truncated_or_prose"
            else:
                key = "invalid_other"
            stats[key] += 1
            examples.setdefault(key, raw[:400])
            continue
        if not parsed:
            stats["empty_list"] += 1
        for p in parsed:
            x1, y1, x2, y2 = p["bbox"]
            areas.append((x2 - x1) * (y2 - y1) / 1e6)
            if p["label"] not in CLASSES:
                stats["unknown_label"] += 1
        n_gt = len(e["objects"])
        if len(parsed) > n_gt + 2:
            stats["many_more_boxes_than_gt"] += 1
    big = sum(a > 0.25 for a in areas)
    return {"counts": dict(stats), "n_pred_boxes": len(areas),
            "pred_area_median": statistics.median(areas) if areas else 0,
            "pred_boxes_over_25pct_frame": big, "examples": examples}


def class_box_sizes(entries):
    sizes = {c: [] for c in CLASSES}
    for e in entries:
        for o in e["objects"]:
            sizes[o["label"]].append(o["yolo"][2] * o["yolo"][3])
    return {c: {"n": len(v), "median_area_pct": 100 * statistics.median(v)} for c, v in sizes.items()}


def reward_hacking(path, n_calls_per_window=10):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    calls = sorted({r["call"] for r in rows})
    windows = []
    for start in range(0, len(calls), n_calls_per_window):
        chunk = set(calls[start:start + n_calls_per_window])
        w = [r for r in rows if r["call"] in chunk]
        windows.append({
            "calls": f"{min(chunk)}-{max(chunk)}", "n": len(w),
            "mean_reward": sum(r["reward"] for r in w) / len(w),
            "valid_rate": sum(r["valid"] for r in w) / len(w),
            "empty_rate": sum(r["empty"] for r in w) / len(w),
            "giant_boxes": sum(r["giant_boxes"] for r in w),
            "mean_pred_minus_gt": sum(r["n_pred"] - r["n_gt"] for r in w) / len(w)})
    return windows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/extra/analysis.json")
    a = ap.parse_args(argv)
    test = read_split("data", "test")
    names = ["e0", "e1", "e2", "e3", "e4_data_0.1", "e5"]
    preds = {n: load_preds(f"work/{n}.jsonl") for n in names}
    rows = {n: per_image_counts(test, preds[n]) for n in names}
    res = {"paired_bootstrap": {f"{b}-{a_}": paired_bootstrap(rows[a_], rows[b]) for a_, b in
                                [("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3"),
                                 ("e4_data_0.1", "e3"), ("e3", "e5")]}}
    res["e0_failures"] = e0_failures(test, preds["e0"])
    res["class_box_sizes_test"] = class_box_sizes(test)
    res["class_box_sizes_train"] = class_box_sizes(read_split("data", "train"))
    # Thang toạ độ sai: chấm output 0-1000 của E3 như thể là pixel (và ngược lại).
    as_pixel = evaluate(test, preds["e3"], coord="pixel", n_boot=200)
    pixel_preds = {}
    for e in test:
        p = parse_output(preds["e3"][e["image_id"]]) or []
        pixel_preds[e["image_id"]] = json.dumps([{"label": x["label"], "bbox_2d": [
            round(x["bbox"][0] * 640 / 1000), round(x["bbox"][1] * 360 / 1000),
            round(x["bbox"][2] * 640 / 1000), round(x["bbox"][3] * 360 / 1000)]} for x in p])
    pixel_as_norm = evaluate(test, pixel_preds, coord="norm1000", n_boot=200)
    pixel_right = evaluate(test, pixel_preds, coord="pixel", n_boot=200)
    res["coord_mismatch_e3"] = {
        "norm1000_scored_as_pixel_f1": as_pixel["micro"]["f1"],
        "converted_to_pixel_scored_as_norm1000_f1": pixel_as_norm["micro"]["f1"],
        "converted_to_pixel_scored_as_pixel_f1": pixel_right["micro"]["f1"]}
    other = Counter()
    for n in names:
        for raw in preds[n].values():
            for p in parse_output(raw) or []:
                if p["label"] not in CLASSES:
                    other[f"{n}:{p['label']}"] += 1
    res["unknown_labels"] = dict(other)
    res["grpo_reward_windows"] = reward_hacking("report/logs/e5_reward_log.jsonl")
    lens = {n: statistics.mean(len(r) for r in preds[n].values()) for n in names}
    res["mean_raw_chars"] = lens
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
