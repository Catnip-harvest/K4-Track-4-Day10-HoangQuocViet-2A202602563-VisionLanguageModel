"""Sinh lab_day10.ipynb: các phần chạy được trên CPU được thực thi tại máy; phần cần GPU
(train, suy luận) đã chạy trên Kaggle T4 qua kaggle/runner.py, notebook nạp lại kết quả của chúng.

python tools/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace lab_day10.ipynb
"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Lab Day 10: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường
Hoàng Quốc Việt · 2A202602563

Mã của tôi nằm trong gói `day10/` (notebook này gọi lại các hàm đó). Phần cần GPU chạy trên
**Kaggle, 2 GPU T4 16 GB** thay cho Colab T4 (cùng loại GPU), bằng `kaggle/runner.py`: mỗi GPU một
hàng đợi thí nghiệm. Các ô dưới đây nạp lại dự đoán và log do Kaggle sinh ra (`work/`, `report/logs/`).
"""),
    md("## 0. Môi trường"),
    code("""
import json, subprocess, sys
from pathlib import Path
print(Path('report/logs/env.txt').read_text(encoding='utf-8'))
print(subprocess.run([sys.executable, '-m', 'pytest', '-q', '--color=no', 'tests', 'day10/tests'],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1])
"""),
    md("## 1. Dữ liệu: split theo phiên quay"),
    code("""
from collections import Counter
from day10.common import read_split
meta = json.loads(Path('data/splits/meta.json').read_text(encoding='utf-8'))
print('dataset_version:', meta['dataset_version'])
sessions = {}
for s in ('train', 'val', 'test'):
    rows = read_split('data', s)
    boxes = Counter(o['label'] for r in rows for o in r['objects'])
    sessions[s] = {r['session'] for r in rows}
    print(f"{s:5s} {len(rows):4d} ảnh  {dict(boxes)}  {len(sessions[s])} phiên")
for a in sessions:
    for b in sessions:
        if a < b:
            print(a, b, 'phiên chung:', len(sessions[a] & sessions[b]))
"""),
    code("""
from PIL import Image
from day10.viz import draw, per_image
rows = read_split('data', 'train')
picks = [rows[i] for i in (0, 120, 300, 600, 900, 1300)]
tiles = []
for k, r in enumerate(picks):
    item = {'entry': r, 'pred': [], 'tp': 0, 'fp': 0, 'fn': 0}
    path = Path(f'report/figures/data_{k}.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    draw('data', item, f"train {r['image_id']}  {len(r['objects'])} box (xanh = nhãn gốc)", path)
    tiles.append(Image.open(path))
w, h = tiles[0].size
grid = Image.new('RGB', (w * 2, h * 3))
for k, t in enumerate(tiles):
    grid.paste(t, ((k % 2) * w, (k // 2) * h))
grid.save('report/figures/data_samples.jpg', quality=85)
for k in range(6):
    Path(f'report/figures/data_{k}.jpg').unlink()
grid.resize((w, int(h * 1.5)))
"""),
    md("## 2. Nhãn: YOLO → JSON thang 0–1000"),
    code("""
from day10.common import PROMPT_NORM1000, objects_to_answer, yolo_to_box
r = rows[0]
answer = objects_to_answer(r['objects'])
print(PROMPT_NORM1000, '\\n')
print(answer)
# Quy ngược về pixel và so với box YOLO gốc. Hai nguồn sai lệch: làm tròn về số nguyên (≤ 0,5 đơn vị),
# và kẹp vào [0, 1000] với vài nhãn gốc tràn ra ngoài khung hình.
worst, clamped, n = 0.0, 0, 0
for row in read_split('data', 'train'):
    for o in row['objects']:
        b = yolo_to_box(o['label'], o['yolo'])['bbox_2d']
        xc, yc, bw, bh = o['yolo']
        ref = [(xc - bw / 2) * 640, (yc - bh / 2) * 360, (xc + bw / 2) * 640, (yc + bh / 2) * 360]
        px = [b[0] * 640 / 1000, b[1] * 360 / 1000, b[2] * 640 / 1000, b[3] * 360 / 1000]
        n += 1
        if min(ref[0], ref[1]) < 0 or ref[2] > 640 or ref[3] > 360:
            clamped += 1
            continue
        worst = max(worst, max(abs(p - q) for p, q in zip(px, ref)))
print(f'{n} box train; {clamped} box nhãn gốc tràn khung bị kẹp vào [0, 1000]')
print(f'các box còn lại: sai lệch lớn nhất khi quy ngược về pixel = {worst:.2f} px (chỉ do làm tròn)')
"""),
    md("## 3. E0: zero-shot"),
    md("""
Đã chạy trên Kaggle: `python -m day10.infer --split test --out work/e0.jsonl` (model gốc, NF4 4-bit,
ViT + `lm_head` giữ fp16). Chấm tại máy bằng trọng tài của đề:
"""),
    code("""
def score(pred, name, out, *meta, coord='norm1000'):
    args = [sys.executable, 'scripts/score.py', '--pred', pred, '--name', name, '--out', out, '--coord', coord]
    for m in meta:
        args += ['--meta', m]
    print(subprocess.run(args, capture_output=True, text=True, encoding='utf-8').stdout)

score('work/e0.jsonl', 'E0', 'report/evidence/e0.json')
raws = [json.loads(l) for l in open('work/e0.jsonl', encoding='utf-8')]
for r in raws[:5]:
    print(r['image_id'], '→', r['raw'][:300].replace('\\n', ' '))
"""),
    md("## 4. Dựng dữ liệu huấn luyện: chat template và mask loss"),
    code("""
checks = json.loads(Path('report/checks.json').read_text(encoding='utf-8'))
ex = checks['example']
print('image_grid_thw', ex['image_grid_thw'], '→ token ảnh:', ex['image_tokens'],
      '(số <|image_pad|> trong input_ids:', ex['image_pad_tokens_in_input_ids'], ')')
print('độ dài chuỗi', ex['seq_len'], '· token có loss', ex['supervised_tokens'],
      f"({ex['supervised_tokens'] / ex['seq_len']:.0%})")
print('giải mã labels != -100:', repr(ex['decoded_supervised']))
print('khớp đáp án:', ex['decoded_supervised'] == ex['answer'] + '<|im_end|>\\n')
print('collate:', checks['collate_shapes'])
"""),
    md("## 5. Setup LoRA / đóng băng cho E1, E2, E3"),
    code("""
for k, v in checks['params'].items():
    print(f"{k:34s} {v:>15,}" if isinstance(v, int) and not isinstance(v, bool) else f'{k:34s} {v}')
print('module LoRA khớp regex:', checks['lora_modules'])
for e in ('e1', 'e2', 'e3'):
    run = json.loads(Path(f'report/logs/{e}_run.json').read_text())
    print(e.upper(), f"trainable={run['trainable_params']:,}  projector={run['trainable_projector_params']:,}  "
          f"LoRA modules={run['matched_lora_modules']}")
"""),
    md("## 6. Vòng huấn luyện"),
    code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4))
for e in ('e1', 'e2', 'e3', 'e4_data_0.1'):
    log = [json.loads(l) for l in open(f'report/logs/{e}_train_log.jsonl')]
    xs = [r['step'] for r in log]
    ys = [r['loss'] for r in log]
    smooth = [sum(ys[max(0, i - 9):i + 1]) / len(ys[max(0, i - 9):i + 1]) for i in range(len(ys))]
    ax.plot(xs, smooth, label=e.upper())
    run = json.loads(Path(f'report/logs/{e}_run.json').read_text())
    print(e.upper(), f"loss đầu {ys[0]:.3f} → cuối {sum(ys[-10:]) / 10:.3f}",
          f"· VRAM đỉnh {run['peak_vram_gb']} GB · {run['train_minutes']} phút")
ax.set_xlabel('bước tối ưu'); ax.set_ylabel('loss (trung bình trượt 10 bước)'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('report/figures/loss_curves.png', dpi=110)
"""),
    md("## 7. Suy luận và chấm E1–E3"),
    code("""
def meta_of(e):
    run = json.loads(Path(f'report/logs/{e}_run.json').read_text())
    return [f"trainable_params={run['trainable_params']}", f"peak_vram_gb={float(run['peak_vram_gb'])}",
            f"train_minutes={float(run['train_minutes'])}"]

for e in ('e1', 'e2', 'e3'):
    score(f'work/{e}.jsonl', e.upper(), f'report/evidence/{e}.json', *meta_of(e))
"""),
    code("""
best = json.loads(Path('report/logs/best.json').read_text())['best']
for name in ('e0', best):
    print(subprocess.run([sys.executable, '-m', 'day10.viz', '--pred', f'work/{name}.jsonl', '--name', name.upper()],
                         capture_output=True, text=True, encoding='utf-8').stdout[:400])
Image.open(f'report/figures/{best}_1_good.jpg')
"""),
    md("## 8. E4 (ablation data_size) và E5 (GRPO)"),
    code("""
print('setup gốc cho E4/E5:', best.upper())
score('work/e4_data_0.1.jsonl', 'E4_data_0.1', 'report/evidence/e4_data_0.1.json', *meta_of('e4_data_0.1'))
# data_size = 1.0 chính là setup gốc (cùng mọi siêu tham số), nên dùng lại dự đoán của nó.
score(f'work/{best}.jsonl', 'E4_data_1.0', 'report/evidence/e4_data_1.0.json', *meta_of(best))
"""),
    code("""
print(subprocess.run([sys.executable, '-m', 'pytest', '-q', '--color=no', 'day10/tests/test_reward.py'],
                     capture_output=True, text=True).stdout.strip().splitlines()[-1])
score('work/e5.jsonl', 'E5', 'report/evidence/e5.json', *meta_of('e5'))
hist = [h for h in json.loads(Path('report/logs/e5_log_history.json').read_text()) if 'reward' in h]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot([h['step'] for h in hist], [h['reward'] for h in hist], label='reward trung bình')
ax.plot([h['step'] for h in hist], [h['completions/mean_length'] / 128 for h in hist], label='độ dài / 128')
ax.set_xlabel('bước GRPO'); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig('report/figures/grpo_reward.png', dpi=110)
"""),
]


def main():
    nb = {"cells": CELLS,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    Path("lab_day10.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
