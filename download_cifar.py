"""
download_cifar.py  (Hugging Face Hub mirror -- thay the ban tai truc tiep
tu cs.toronto.edu)
=====================================================================

LY DO DOI NGUON:
Ban goc tai tu https://www.cs.toronto.edu/~kriz/cifar-*.tar.gz. Server
nay tu dong REDIRECT sang http://cave.cs.toronto.edu:80/... (mot subdomain
phu, HTTP thuong, cong 80) -- host nay thuong KHONG ket noi duoc tu mang
cloud/datacenter (Modal, AWS, GCP, ...) -> ConnectTimeout du da fallback
tai tuan tu, vi ca 2 duong (HEAD probe lan GET that) deu di qua cung 1
redirect chet.

Ban nay tai tu Hugging Face Hub (dataset "uoft-cs/cifar10" /
"uoft-cs/cifar100", dang Parquet), qua CDN HTTPS chuan cua HF -- on dinh
hon nhieu tu moi loai mang cloud, va thu vien `datasets` da tu lo
retry/resume/backoff nen KHONG can tu viet lai logic download/Range/thread
nhu ban cu.

YEU CAU THEM: pip install datasets huggingface_hub

Cau truc output (GIONG HET ban cu, cac script train doc thang duoc,
khong can sua GWGA_single_input.py / standard_KD.py):
    dataset/cifar-10/
        train_images.npy   (50000, 32, 32, 3) uint8
        train_labels.npy   (50000,)           int64
        test_images.npy    (10000, 32, 32, 3) uint8
        test_labels.npy    (10000,)           int64
        classes.txt         ten 10 class

    dataset/cifar-100/
        train_images.npy         (50000, 32, 32, 3) uint8
        train_labels.npy         (50000,)  fine label, int64
        train_coarse_labels.npy  (50000,)  coarse label, int64
        test_images.npy / test_labels.npy / test_coarse_labels.npy
        fine_classes.txt     ten 100 class
        coarse_classes.txt   ten 20 superclass

Su dung:
    python download_cifar.py                  # tai ca 2 bo
    python download_cifar.py --which cifar10   # chi tai CIFAR-10
    python download_cifar.py --which cifar100  # chi tai CIFAR-100

Neu mang cua ban CHAN LUON huggingface.co (hiem, nhung co the xay ra o
mot so mang doanh nghiep/dai hoc), dat bien moi truong HF_ENDPOINT tro
toi mirror khac truoc khi chay, vi du:
    export HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# =========================================================================
# DUONG DAN OUTPUT (giong het ban cu)
# =========================================================================
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
CIFAR10_DIR = DATASET_DIR / "cifar-10"
CIFAR100_DIR = DATASET_DIR / "cifar-100"

CIFAR10_HF_REPO = "uoft-cs/cifar10"
CIFAR100_HF_REPO = "uoft-cs/cifar100"


# =========================================================================
# TIEN ICH
# =========================================================================
def _decode_images(split_dataset, image_col: str = "img") -> np.ndarray:
    """Cot 'img' cua HF dataset la PIL.Image (lazy-decode) -> stack thanh
    (N, 32, 32, 3) uint8. Ep .convert('RGB') de phong truong hop anh
    grayscale/RGBA le te trong metadata bi sai mode."""
    images = np.stack(
        [np.array(ex[image_col].convert("RGB"), dtype=np.uint8)
         for ex in tqdm(split_dataset, desc="  decoding images")],
        axis=0,
    )
    return images


def _npy_exists(out_dir: Path, split: str, extra_names: list[str] = ()) -> bool:
    names = [f"{split}_images.npy", f"{split}_labels.npy"]
    names += [f"{split}_{n}.npy" for n in extra_names]
    return all((out_dir / n).exists() for n in names)


# =========================================================================
# CIFAR-10
# =========================================================================
def download_cifar10(out_dir: Path) -> None:
    from datasets import load_dataset  # import lazy de --which cifar100 khong bat buoc datasets neu chi can 1 bo

    if _npy_exists(out_dir, "train") and _npy_exists(out_dir, "test") and (out_dir / "classes.txt").exists():
        print(f"[skip] CIFAR-10 da co san trong {out_dir}.")
        return

    print(f"\n[HF] Loading {CIFAR10_HF_REPO} tu Hugging Face Hub ...")
    ds = load_dataset(CIFAR10_HF_REPO)   # tai + cache parquet qua HTTPS CDN cua HF

    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = ds["train"].features["label"].names   # dung thu tu voi pickle goc
    (out_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    for split in ["train", "test"]:
        print(f"[CIFAR-10] Decoding split '{split}' ({len(ds[split])} anh) ...")
        images = _decode_images(ds[split])
        labels = np.array(ds[split]["label"], dtype=np.int64)
        np.save(out_dir / f"{split}_images.npy", images)
        np.save(out_dir / f"{split}_labels.npy", labels)
        print(f"[saved] {split}: images {images.shape}, labels {labels.shape} -> {out_dir}")


# =========================================================================
# CIFAR-100
# =========================================================================
def download_cifar100(out_dir: Path) -> None:
    from datasets import load_dataset

    if (_npy_exists(out_dir, "train", ["coarse_labels"]) and
            _npy_exists(out_dir, "test", ["coarse_labels"]) and
            (out_dir / "fine_classes.txt").exists()):
        print(f"[skip] CIFAR-100 da co san trong {out_dir}.")
        return

    print(f"\n[HF] Loading {CIFAR100_HF_REPO} tu Hugging Face Hub ...")
    ds = load_dataset(CIFAR100_HF_REPO)

    out_dir.mkdir(parents=True, exist_ok=True)
    fine_names = ds["train"].features["fine_label"].names
    coarse_names = ds["train"].features["coarse_label"].names
    (out_dir / "fine_classes.txt").write_text("\n".join(fine_names), encoding="utf-8")
    (out_dir / "coarse_classes.txt").write_text("\n".join(coarse_names), encoding="utf-8")

    for split in ["train", "test"]:
        print(f"[CIFAR-100] Decoding split '{split}' ({len(ds[split])} anh) ...")
        images = _decode_images(ds[split])
        fine_labels = np.array(ds[split]["fine_label"], dtype=np.int64)
        coarse_labels = np.array(ds[split]["coarse_label"], dtype=np.int64)
        np.save(out_dir / f"{split}_images.npy", images)
        np.save(out_dir / f"{split}_labels.npy", fine_labels)
        np.save(out_dir / f"{split}_coarse_labels.npy", coarse_labels)
        print(f"[saved] {split}: images {images.shape}, fine_labels {fine_labels.shape}, "
              f"coarse_labels {coarse_labels.shape} -> {out_dir}")


# =========================================================================
# MAIN
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download va chuan bi CIFAR-10 / CIFAR-100 tu Hugging Face Hub "
                     "(mirror on dinh hon cs.toronto.edu tren mang cloud)")
    parser.add_argument("--which", choices=["cifar10", "cifar100", "both"], default="both")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Thieu thu vien 'datasets'. Cai dat bang:\n"
            "    pip install datasets huggingface_hub\n"
            "roi chay lai script nay."
        )

    if args.which in ("cifar10", "both"):
        print("\n=== CIFAR-10 ===")
        download_cifar10(CIFAR10_DIR)
    if args.which in ("cifar100", "both"):
        print("\n=== CIFAR-100 ===")
        download_cifar100(CIFAR100_DIR)

    print("\nXong. Du lieu da duoc luu trong thu muc dataset/.")


if __name__ == "__main__":
    main()