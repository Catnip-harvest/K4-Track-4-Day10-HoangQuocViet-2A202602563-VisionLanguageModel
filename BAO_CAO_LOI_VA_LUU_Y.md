# Báo cáo lỗi và lưu ý kỹ thuật

Các bẫy tôi gặp thật trong lab này, kèm triệu chứng, nguyên nhân và cách xử lý.

| # | Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|---|
| 1 | E3 dừng với `ValueError: Attempting to unscale FP16 gradients`; số tham số projector huấn luyện là 54,579,200 thay vì 109,105,152 | transformers 5.19 so khớp `llm_int8_skip_modules` bằng `re.match` từ đầu tên hoặc khớp đuôi, nên `"visual"` không khớp `model.visual.*`. Toàn bộ ViT và projector bị lượng tử hoá 4-bit (tensor 4-bit đóng gói có `numel` bằng một nửa) | Dùng `["visual", "model.visual", "lm_head"]` và `assert_vit_not_quantized()` trong `day10/common.py` để dừng ngay khi nạp model nếu ViT còn `Linear4bit` |
| 2 | E3 lỗi `ImportError: Found an incompatible version of torchao` khi gắn LoRA | Kaggle cài sẵn torchao 0.10; peft 0.21.2 kiểm tra torchao khi bọc `nn.Linear` thường (ViT fp16) | `pip uninstall -y torchao` trong `kaggle/runner.py` |
| 3 | `GRPOTrainer` đưa tham số LoRA về bf16 | TRL làm vậy với model lượng tử hoá; T4 không hỗ trợ bf16 và GradScaler fp16 cần tham số fp32 | Sau khi tạo trainer, đưa tham số huấn luyện về fp32 và đặt `fp16=True, bf16=False` (`day10/grpo.py`) |
| 4 | Completion GRPO đúng nhưng bị cắt thành JSON hỏng | `max_completion_length=128`, trong khi đáp án của ảnh nhiều box dài hơn (≈ 27 token/box) | Chỉ dùng 1.280/1.422 ảnh có đáp án vừa 128 token |
| 5 | Sau khi clone trên Windows, mọi file của đề hiện là "modified" | `core.autocrlf=true` chuyển LF sang CRLF; nếu commit sẽ làm đổi `scripts/` và `tests/` (−10 điểm) | `git config core.autocrlf false` rồi `git checkout -- .` (chỉ khác ký tự xuống dòng) |
| 6 | E0 hỏng JSON 16% | Model gốc lặp box đến hết 256 token (45/50 output hỏng) | Không sửa (là kết quả của E0); sau finetune `parse_valid_rate` ≥ 0,987 |
| 7 | GradScaler bỏ qua vài bước đầu (`grad_norm` NaN ở bước 1) | Hệ số scale ban đầu 65536 làm gradient fp16 tràn; GradScaler bỏ bước đó và giảm scale | Hành vi bình thường, loss vẫn hữu hạn; vòng train dừng nếu *loss* không hữu hạn |

## Lưu ý khi chạy lại

- GPU: Kaggle T4 x2 (`python kaggle/build.py --stage full --ref <commit> --dir <thư mục>` rồi
  `kaggle kernels push -p <thư mục> --accelerator NvidiaTeslaT4`). Dữ liệu gắn bằng `dataset_sources`, không cần khoá trong mã.
- Chấm điểm tại máy: tải `out/` của kernel về `work/`, rồi chạy các ô của `lab_day10.ipynb` (gọi `scripts/score.py` với `--meta` lấy từ `report/logs/*_run.json`).
- `e4_data_1.0.json` là dự đoán của E3 chấm lại dưới tên E4: data_size 1.0 chính là setup gốc, mọi siêu tham số giống hệt.
- Trên Windows, chạy công cụ của đề với `PYTHONUTF8=1` để tránh lỗi bảng mã cp1252.
