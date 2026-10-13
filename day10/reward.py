"""Bước 9: reward IoU kiểm chứng được cho GRPO (Python thuần, test được không cần GPU)."""
from scripts.score import iou_xyxy, parse_output

IOU_THRESHOLD = 0.5


def gt_boxes_from_objects(objects):
    """Nhãn YOLO -> [{"label", "bbox"}] thang 0-1000, cùng quy ước với scripts/score.py."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": ((xc - w / 2) * 1000, (yc - h / 2) * 1000,
                                                  (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def matched_iou_sum(gt, pred):
    """Ghép tham lam theo IoU giảm dần, cùng lớp, chỉ nhận IoU >= 0.5. Trả tổng IoU các cặp."""
    total = 0.0
    for label in {g["label"] for g in gt} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt if x["label"] == label]
        p = [x["bbox"] for x in pred if x["label"] == label]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
                       reverse=True)
        used_g, used_p = set(), set()
        for value, i, j in pairs:
            if value < IOU_THRESHOLD:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total += value
    return total


def compute_reward(completion, gt):
    pred = parse_output(completion)
    if pred is None:
        return 0.0  # JSON hỏng
    if not gt and not pred:
        return 1.0  # cả hai cùng rỗng
    # Mẫu số max(số gt, số pred): box bịa thêm hay box bỏ sót đều kéo reward xuống.
    return 0.2 + 0.8 * matched_iou_sum(gt, pred) / max(len(gt), len(pred), 1)
