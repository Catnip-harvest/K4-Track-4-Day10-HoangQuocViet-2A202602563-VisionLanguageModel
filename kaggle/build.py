"""Dựng thư mục kernel Kaggle từ kaggle/runner.py.

python kaggle/build.py --stage smoke --ref <commit> --dir <thư mục tạm>
kaggle kernels push -p <thư mục tạm> --accelerator NvidiaTeslaT4
"""
import argparse
import json
from pathlib import Path

DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "full"])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--user", default="viethwang3i")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    src = (Path(__file__).parent / "runner.py").read_text(encoding="utf-8")
    (d / "runner.py").write_text(src.replace("__STAGE__", a.stage).replace("__REF__", a.ref),
                                 encoding="utf-8")
    slug = f"k4-day10-vlm-{a.stage}"
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": "runner.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [DATASET],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(d, slug)


if __name__ == "__main__":
    main()
