import json

import pytest

from day10.reward import compute_reward, gt_boxes_from_objects

GT = [{"label": "pothole", "bbox": (100, 100, 200, 200)},
      {"label": "crack", "bbox": (500, 500, 700, 600)}]


def answer(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


def test_perfect_prediction_scores_one():
    assert compute_reward(answer(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))),
                          GT) == pytest.approx(1.0)


def test_broken_json_scores_zero():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert compute_reward("Based on the image, there is a pothole.", GT) == 0.0


def test_wrong_label_gets_only_the_format_reward():
    pred = answer(("manhole", (100, 100, 200, 200)), ("manhole", (500, 500, 700, 600)))
    assert compute_reward(pred, GT) == pytest.approx(0.2)


def test_extra_boxes_are_penalised_through_the_denominator():
    one_gt = GT[:1]
    pred = answer(("pothole", (100, 100, 200, 200)), ("pothole", (600, 600, 700, 700)))
    assert compute_reward(pred, one_gt) == pytest.approx(0.2 + 0.8 * 1.0 / 2)  # 0.6


def test_both_empty_scores_one():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_on_damaged_image_gets_only_format_reward():
    assert compute_reward("[]", GT) == pytest.approx(0.2)


def test_giant_box_does_not_pay():
    assert compute_reward(answer(("pothole", (0, 0, 1000, 1000))), GT) == pytest.approx(0.2)


def test_partial_overlap_pays_its_iou():
    # dịch box 20 đơn vị: IoU = 80*100 / (2*100*100 - 80*100) = 0.6667
    pred = answer(("pothole", (120, 100, 220, 200)))
    assert compute_reward(pred, GT[:1]) == pytest.approx(0.2 + 0.8 * (8000 / 12000))


def test_markdown_fence_is_accepted_like_the_scorer():
    fenced = "```json\n" + answer(("pothole", (100, 100, 200, 200))) + "\n```"
    assert compute_reward(fenced, GT[:1]) == pytest.approx(1.0)


def test_gt_conversion_matches_norm1000_scale():
    boxes = gt_boxes_from_objects([{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.1]}])
    assert boxes[0]["bbox"] == pytest.approx((400, 450, 600, 550))
