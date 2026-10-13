# Checklist nộp bài Lab Day 10

- [x] `REPORT.md` đã điền họ tên, `dataset_version` (d10-21e4070301), bảng tổng hợp E0–E5 và 10 câu hỏi.
- [x] Không còn chuỗi `<điền` trong `REPORT.md`.
- [x] Có ≥ 6 ảnh minh hoạ box dự đoán và box đúng: `report/figures/e3_*.jpg` (7 ảnh), `e0_*.jpg` (7 ảnh), `data_samples.jpg`.
- [x] Không track file cấm: không có `data/`, `work/`, `.env`, `*.safetensors`, `*.bin`, `*.pt` (adapter và projector chỉ nằm trên Kaggle).
- [x] `git rev-list --count HEAD` ≥ 5, mỗi commit một bước.
- [x] `python scripts/check_submission.py`: 0 FAIL, 0 WARN.
- [x] `scripts/`, `tests/`, `configs/` giữ nguyên bản gốc của đề (`git diff upstream/main -- scripts tests configs` rỗng).
- [x] `python -m pytest -q`: 67 passed, 1 skipped, 1 failed. Test hỏng là `test_report_template_has_student_markers_and_ten_questions`:
  nó kiểm tra *mẫu* REPORT.md còn chuỗi `<điền`, nên sẽ hỏng khi báo cáo đã điền (điều `check_submission.py` bắt buộc).
  Trước khi điền báo cáo: 68 passed, 1 skipped. `python -m pytest -q day10/tests`: 10 passed (reward E5).
- [x] Repo ở chế độ Public.
