"""
data_visualize.py

Vẽ lưới ảnh 10x10:
    - 5 dòng đầu : ảnh ngẫu nhiên lấy từ CIFAR-10 (dataset/cifar-10/train_images.npy)
    - 5 dòng cuối: ảnh ngẫu nhiên lấy từ CIFAR-100 (dataset/cifar-100/train_images.npy)

Không đọc/label gì cả, chỉ visualize ảnh.
Kết quả được lưu thành 1 file PNG duy nhất.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Cấu hình ----------------------------- #
SEED = 42
N_ROWS_PER_DATASET = 5      # 5 dòng cho mỗi dataset
N_COLS = 10                 # 10 ảnh mỗi dòng -> tổng 10x10
TOTAL_ROWS = N_ROWS_PER_DATASET * 2

CIFAR10_IMAGES_PATH = os.path.join("dataset", "cifar-10", "train_images.npy")
CIFAR100_IMAGES_PATH = os.path.join("dataset", "cifar-100", "train_images.npy")

OUTPUT_DIR = "figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "cifar10_cifar100_grid.png")


def load_images(npy_path: str) -> np.ndarray:
    """Load file .npy chứa ảnh, trả về mảng shape (N, H, W, C)."""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Không tìm thấy file: {npy_path}")
    images = np.load(npy_path)
    return images


def sample_images(images: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Lấy ngẫu nhiên n ảnh từ mảng images (không lặp lại nếu đủ số lượng)."""
    n_available = images.shape[0]
    replace = n_available < n
    idx = rng.choice(n_available, size=n, replace=replace)
    return images[idx]


def build_grid_figure(cifar10_images: np.ndarray, cifar100_images: np.ndarray) -> plt.Figure:
    """Tạo figure dạng lưới 10x10 (5 dòng CIFAR-10 + 5 dòng CIFAR-100)."""
    fig, axes = plt.subplots(
        TOTAL_ROWS, N_COLS,
        figsize=(N_COLS * 1.2, TOTAL_ROWS * 1.2),
    )

    n_top = N_ROWS_PER_DATASET * N_COLS  # số ảnh CIFAR-10 cần
    cifar10_flat = cifar10_images[:n_top]
    cifar100_flat = cifar100_images[:n_top]

    for row in range(TOTAL_ROWS):
        for col in range(N_COLS):
            ax = axes[row, col]
            flat_idx = row * N_COLS + col

            if row < N_ROWS_PER_DATASET:
                img = cifar10_flat[flat_idx]
            else:
                top_offset = flat_idx - n_top
                img = cifar100_flat[top_offset]

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Nhãn mô tả 2 khối dữ liệu bên trái
    axes[0, 0].set_ylabel("CIFAR-10", fontsize=11, rotation=90, labelpad=10)
    axes[N_ROWS_PER_DATASET, 0].set_ylabel("CIFAR-100", fontsize=11, rotation=90, labelpad=10)
    axes[0, 0].yaxis.set_visible(True)
    axes[N_ROWS_PER_DATASET, 0].yaxis.set_visible(True)

    fig.suptitle("CIFAR-10 & CIFAR-100", fontsize=30)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    rng = np.random.default_rng(SEED)

    cifar10_images = load_images(CIFAR10_IMAGES_PATH)
    cifar100_images = load_images(CIFAR100_IMAGES_PATH)

    n_needed = N_ROWS_PER_DATASET * N_COLS  # 50 ảnh mỗi dataset
    cifar10_sample = sample_images(cifar10_images, n_needed, rng)
    cifar100_sample = sample_images(cifar100_images, n_needed, rng)

    fig = build_grid_figure(cifar10_sample, cifar100_sample)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã lưu ảnh visualize tại: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()