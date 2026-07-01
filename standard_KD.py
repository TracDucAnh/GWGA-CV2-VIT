"""
standard_KD.py  (ViT setup)
============================
Standard Knowledge Distillation (Hinton, Vinyals & Dean, 2015,
"Distilling the Knowledge in a Neural Network") -- phien ban BASELINE
THUAN, KHONG con Bayesian Last Layer (BLL/ELBO), KHONG con
Gromov-Wasserstein (GW). Day la mot pipeline doc lap, tu-chua-du-lieu:

    L_KD = CE(y_true, student) + alpha * T^2 * KL( softmax(teacher_logits / T)
                                                    || softmax(student_logits / T) )

trong do T la temperature va alpha la trong so cua soft-target loss (xem
hinton_kd_loss() ben duoi va Config.kd_temperature / Config.kd_alpha_max).
KHONG co thanh phan Bayesian/posterior/particle nao, KHONG co ELBO/KL-to-prior,
KHONG co Gromov-Wasserstein structural alignment. Teacher va student deu la
classifier deterministic thong thuong (backbone + 1 Linear head).

Kien truc & du lieu -- Vision Transformer:
  - Teacher  : ViT-Large/16 (timm, pretrained ImageNet) + Linear head moi
               khoi tao. Backbone pretrained co the duoc finetune nhe hoac
               freeze. Huan luyen bang CE thong thuong (khong ELBO).
               So epoch fit teacher = so epoch distill student.
  - Student  : ViT-Small/16 (timm, KHONG pretrained -- random init) +
               Linear head, distill tu dau bang CE + standard KD
               soft-target loss (Hinton 2015).
  - Train set       : CIFAR-100 (num_labels = 100), anh duoc UPSIZE tu
                       32x32 len 224x224 (img_size) de khop input cua
                       ViT-Large/Small patch16/224.
  - Eval (in-dist)  : CIFAR-100 test split.
  - OOD test        : CIFAR-10 (AUROC / OOD detection, KHONG dung de train).

Standard KD (Hinton 2015): moi model (teacher/student) forward 1 LAN, cho
ra 1 vector logits/sample [B, C] (KHONG con K particle nhu ban BLL). Loss
soft-target la KL divergence giua softmax(teacher/T) va softmax(student/T),
nhan T^2 -- xem hinton_kd_loss().

So voi ban CNN/ResNet truoc -- diem khac biet quan trong (giu nguyen tu
ban goc, khong lien quan BLL/GW):
  ViT (timm) dung LayerNorm, KHONG dung BatchNorm. LayerNorm:
    - KHONG co running statistics (buffer) nhu BatchNorm -- thong ke duoc
      tinh TUC THOI tren chinh moi forward pass, bat ke dang train hay
      eval. Vi vay loss landscape KHONG can bat ky context-manager dac
      biet nao de "tam thoi chuyen sang dung batch statistics" nhu ban
      CNN -- van de do KHONG TON TAI voi LayerNorm.
    - Affine weight/bias cua LayerNorm la vector 1 chieu (1 gia tri/dim),
      giong Conv/Linear bias -- da duoc dieu kien "d.dim() <= 1" trong
      filter_normalize_direction() zero-out tu dong (xem Li et al., 2018).
  -> Ket qua: landscape_named_parameters() va compute_loss_landscape()
     don gian hon han ban CNN (khong can quet/khoi-phuc trang thai BN),
     va o ban nay con don gian hon nua vi KHONG con tham so BLL
     (log_sigma) nao can loai tru.

Cac diem giu nguyen (logic pipeline khong doi so voi ban truoc, chi bo
BLL/GW va cac thanh phan phu thuoc vao chung):
  - 1 Config dataclass duy nhat.
  - Loss landscape duoc ve TRUOC khi train (random init ca 2 model) va
    SAU khi distill xong (filter-normalized random directions, 1x2/2x2 grid,
    thang mau/truc LINEAR -- khong dung log scale).
  - UMAP duoc ve moi `umap_every_n_steps` cho CA teacher lan student trong
    luc distill (cung 1 probe batch co dinh xuyen suot training) -- gio
    la UMAP cua 1 vector logits/sample (khong con "particle spread").
  - 2-phase schedule (task-only -> distillation ramp) cho student. (Bo
    phase "posterior" vi khong con ELBO/KL-to-prior de warmup.)
  - ECE, Hessian-trace sharpness (Hutchinson estimator), OOD AUROC
    (CIFAR-100 ID vs CIFAR-10 OOD) deu giu nguyen VE MAT Y NGHIA, chi
    khong con buoc "trung binh qua K particle" (model deterministic ->
    1 forward duy nhat).
  - Moi figure quan trong deu luu kem du lieu tho (.npz/.json) canh file
    .png, va phan ve duoc tach rieng thanh plot_xxx(data, style) de co the
    doc lai va ve lai voi style khac ma khong can tinh toan lai
    (replot_all_from_saved() o cuoi file).
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import math
import json
import time
import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import timm
import torchvision.transforms as T
from PIL import Image

warnings.filterwarnings("ignore")


# =========================================================================
# DATASET PATHS (du lieu DA TAI VE SAN, nam trong thu muc `dataset/` o
# project root -- KHONG bao gio tai lai/download trong code train).
# =========================================================================
#   GWGA-CV2-VIT/                  <-- project root, file nay nam TRUC TIEP o day
#     dataset/
#       cifar-10/   {train,test}_images.npy, {train,test}_labels.npy, classes.txt
#       cifar-100/  {train,test}_images.npy, {train,test}_labels.npy,
#                   {train,test}_coarse_labels.npy, fine_classes.txt, coarse_classes.txt
#     checkpoints/
#     figures/
#     standard_KD.py              <-- file nay (Hinton 2015 baseline, thuan)
#
# _THIS_DIR DA CHINH LA project root, khong can _PROJECT_ROOT.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET_ROOT = os.path.join(_THIS_DIR, "dataset")
_DEFAULT_CIFAR10_DIR = os.path.join(_DEFAULT_DATASET_ROOT, "cifar-10")
_DEFAULT_CIFAR100_DIR = os.path.join(_DEFAULT_DATASET_ROOT, "cifar-100")


# =========================================================================
# CONFIG  (1 dataclass duy nhat, tat ca hyperparams o day)
# =========================================================================

@dataclass
class Config:
    # ── Models (ViT, qua timm.create_model) ─────────────────────────────
    teacher_name: str = "vit_large_patch16_224"
    student_name: str = "vit_small_patch16_224"
    teacher_pretrained: bool = True     # teacher: load checkpoint pretrained (ImageNet)
    student_pretrained: bool = False    # student: distill tu dau (random init)
    img_size: int = 224                 # upsize CIFAR (32x32) len 224x224 -- input chuan cua patch16/224

    # ── Data ──────────────────────────────────────────────────────────────
    # Train + eval-in-distribution: CIFAR-100. OOD test: CIFAR-10.
    # Du lieu DA duoc tai ve san duoi dang .npy trong dataset/cifar-10 va
    # dataset/cifar-100 -- KHONG download=True o bat ky dau (xem
    # NpyImageDataset / build_*_loader(s) ben duoi).
    train_dataset: str = "cifar100"
    id_eval_dataset: str = "cifar100"
    ood_dataset: str = "cifar10"
    cifar10_dir: str = _DEFAULT_CIFAR10_DIR
    cifar100_dir: str = _DEFAULT_CIFAR100_DIR
    num_labels: int = 100
    num_workers: int = 4

    # ── Teacher fit (tren pretrained backbone, CE thong thuong) ──────────
    # YEU CAU: so epoch fit teacher = so epoch distill student -> gan o
    # ngay sau khi tao CFG (xem duoi class Config).
    teacher_num_epochs: int = 10
    teacher_finetune_backbone: bool = True
    teacher_backbone_lr_mult: float = 0.1   # backbone lr = learning_rate * mult

    # ── Student distillation ─────────────────────────────────────────────
    student_num_epochs: int = 10
    phase1_frac: float = 0.3      # ti le epoch DAU chi train CE (khong KD)
    kd_alpha_max: float = 1.0     # trong so toi da cua soft-target KD loss (sau phase1)

    # ── Optimization ──────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision_dtype: torch.dtype = torch.bfloat16

    batch_size: int = 64
    distill_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True   # timm ViT ho tro set_grad_checkpointing()

    # ── Standard Knowledge Distillation (Hinton, Vinyals & Dean, 2015) ──
    # L_soft = T^2 * KL( softmax(teacher/T) || softmax(student/T) ), tinh
    # TRUC TIEP giua vector logits cua teacher va vector logits cua
    # student -- xem hinton_kd_loss(). T cao hon -> phan phoi "mem" hon,
    # lo nhieu thong tin "dark knowledge" giua cac lop sai hon (dung gia
    # tri T=4 theo de xuat goc cua Hinton et al., 2015 cho cac bai toan
    # classification nhieu lop).
    kd_temperature: float = 4.0

    # ── UMAP probe ────────────────────────────────────────────────────────
    umap_every_n_steps: int = 100
    umap_probe_samples: int = 256
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"
    umap_n_epochs: int = 500
    umap_seed: int = 42

    # ── Loss landscape ───────────────────────────────────────────────────
    landscape_grid_size: int = 15
    landscape_alpha_range: Tuple[float, float] = (-2.0, 2.0)
    landscape_beta_range: Tuple[float, float] = (-2.0, 2.0)
    landscape_eval_batches: int = 10
    landscape_eval_batch_size: int = 16
    landscape_seed: int = 42
    # Khong can context-manager rieng cho LayerNorm nhu BatchNorm o ban
    # CNN -- LayerNorm khong co running stats. Cung khong con tham so BLL
    # (log_sigma) nao can loai tru khoi landscape_named_parameters().

    # ── ECE (Expected Calibration Error) ─────────────────────────────────
    ece_num_bins: int = 15

    # ── Hessian-trace sharpness (Hutchinson estimator) ───────────────────
    hessian_num_hutchinson_samples: int = 10
    hessian_eval_batches: int = 5
    hessian_seed: int = 777

    # ── OOD evaluation (CIFAR-10 dung lam OOD cho model train tren CIFAR-100) ─
    ood_score_type: str = "predictive_entropy"

    current_dir = Path(__file__).resolve().parent
    base_output_dir = current_dir / "output_vit_cifar_kd_standard"

    output_dir = str(base_output_dir)
    checkpoint_dir = str(base_output_dir / "checkpoints")
    figure_dir = str(base_output_dir / "figures")
    figure_data_dir = str(base_output_dir / "figure_data")
    log_file = str(base_output_dir / "train_log.jsonl")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(exist_ok=True)
    Path(figure_dir).mkdir(exist_ok=True)
    Path(figure_data_dir).mkdir(exist_ok=True)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1234

    run_pre_landscape: bool = True
    run_teacher_fit: bool = True
    run_distillation: bool = True
    run_post_landscape: bool = True
    run_metric_curves: bool = True
    run_ood_eval: bool = True


CFG = Config()
# YEU CAU 1: teacher duoc fit voi so epoch BANG so epoch distill student.
CFG.teacher_num_epochs = CFG.student_num_epochs

TEACHER_KEY = "ViT-Large (teacher)"
STUDENT_KEY = "ViT-Small (student, KD)"

_CMAP_100 = plt.get_cmap("tab20")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(record)


def save_figure_data(cfg: Config, name: str, payload: dict):
    """Luu du lieu tho dung de ve 1 figure (.npz cho mang so + .json cho
    metadata), de co the goi lai plot_xxx(...) voi PlotStyle khac ma
    khong can tinh toan lai (khong can GPU / khong can forward model)."""
    os.makedirs(cfg.figure_data_dir, exist_ok=True)
    arrays = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    meta   = {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}
    np.savez(os.path.join(cfg.figure_data_dir, f"{name}.npz"), **arrays)
    with open(os.path.join(cfg.figure_data_dir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_figure_data(cfg: Config, name: str) -> dict:
    arrays = dict(np.load(os.path.join(cfg.figure_data_dir, f"{name}.npz")))
    meta_path = os.path.join(cfg.figure_data_dir, f"{name}.meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {**arrays, **meta}


# =========================================================================
# PLOT STYLE
# =========================================================================

@dataclass
class PlotStyle:
    figsize_single: Tuple[float, float] = (8.0, 6.0)
    figsize_wide:   Tuple[float, float] = (14.0, 5.5)
    figsize_grid2x2: Tuple[float, float] = (14.0, 11.0)
    dpi: int = 150
    title_fontsize: int = 14
    subtitle_fontsize: int = 12
    label_fontsize: int = 11
    legend_fontsize: int = 9
    tick_fontsize: int = 9
    marker_size: float = 6.0
    line_width: float = 2.0
    teacher_color: str = "tab:blue"
    student_color: str = "tab:orange"
    band_alpha: float = 0.18
    grid_alpha: float = 0.3
    cmap_name: str = "viridis"


DEFAULT_STYLE = PlotStyle()


# =========================================================================
# DATA: CIFAR-100 (train/eval-in-dist) + CIFAR-10 (OOD)
# =========================================================================

def build_image_transforms(cfg: Config, train: bool) -> T.Compose:
    """Upsize CIFAR (32x32) len img_size (224, mac dinh) cho ViT, normalize
    theo thong ke ImageNet (chuan khi dung backbone pretrained cua timm)."""
    ops = [T.Resize((cfg.img_size, cfg.img_size))]
    if train:
        ops += [T.RandomHorizontalFlip(p=0.5)]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(ops)


class NpyImageDataset(Dataset):
    """
    Dataset CIFAR-10/100 doc TRUC TIEP tu cac file .npy DA CO SAN tren dia
    (dataset/cifar-10/ hoac dataset/cifar-100/). KHONG bao gio goi mang --
    neu thieu file se bao loi ro rang (FileNotFoundError) thay vi tu dong
    tai ve.

    Gia dinh format (chuan khi dump tu torchvision.datasets.CIFARxx, vd
    `np.save(path, dataset.data)`):
      - images_path : uint8, shape [N, 32, 32, 3] (HWC, RGB). Tu dong
                      chuyen [N, 3, 32, 32] (CHW) -> HWC neu phat hien.
      - labels_path : int, shape [N].
    """
    def __init__(self, images_path: str, labels_path: str, transform=None):
        if not os.path.isfile(images_path):
            raise FileNotFoundError(
                f"[NpyImageDataset] Khong tim thay file anh: {images_path}\n"
                f"  -> Du lieu duoc gia dinh la DA TAI VE SAN (xem cfg.cifar10_dir / "
                f"cfg.cifar100_dir). File nay KHONG tu dong tai ve du lieu."
            )
        if not os.path.isfile(labels_path):
            raise FileNotFoundError(f"[NpyImageDataset] Khong tim thay file nhan: {labels_path}")

        self.images = np.load(images_path)
        self.labels = np.load(labels_path).astype(np.int64).reshape(-1)
        if self.images.ndim != 4:
            raise ValueError(
                f"[NpyImageDataset] Mong doi anh 4 chieu [N,H,W,C] hoac [N,C,H,W], "
                f"nhung nhan duoc shape={self.images.shape} tu {images_path}")
        if self.images.shape[1] == 3 and self.images.shape[-1] != 3:
            self.images = self.images.transpose(0, 2, 3, 1)
        if self.images.dtype != np.uint8:
            self.images = self.images.astype(np.uint8)
        assert len(self.images) == len(self.labels), (
            f"[NpyImageDataset] So anh ({len(self.images)}) != so nhan ({len(self.labels)}) "
            f"trong {images_path} / {labels_path}")

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_cifar100_loaders(cfg: Config, batch_size: int):
    """Doc CIFAR-100 train/test TU .npy CO SAN trong cfg.cifar100_dir (khong tai lai)."""
    train_tf = build_image_transforms(cfg, train=True)
    eval_tf  = build_image_transforms(cfg, train=False)
    train_set = NpyImageDataset(
        os.path.join(cfg.cifar100_dir, "train_images.npy"),
        os.path.join(cfg.cifar100_dir, "train_labels.npy"),
        transform=train_tf)
    eval_set = NpyImageDataset(
        os.path.join(cfg.cifar100_dir, "test_images.npy"),
        os.path.join(cfg.cifar100_dir, "test_labels.npy"),
        transform=eval_tf)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, eval_loader


def build_cifar10_ood_loader(cfg: Config, batch_size: int):
    """CIFAR-10 dung LAM OOD test set cho model train tren CIFAR-100, doc TU
    .npy CO SAN trong cfg.cifar10_dir (khong tai lai)."""
    eval_tf = build_image_transforms(cfg, train=False)
    ood_set = NpyImageDataset(
        os.path.join(cfg.cifar10_dir, "test_images.npy"),
        os.path.join(cfg.cifar10_dir, "test_labels.npy"),
        transform=eval_tf)
    return DataLoader(
        ood_set, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )


class _FixedIndexSubset(Dataset):
    """Subset co dinh (theo seed) dung lam probe batch cho UMAP / landscape."""
    def __init__(self, base_dataset, indices: List[int]):
        self.base = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


def build_umap_probe_batch(cfg: Config, dataset_name: str = "cifar100") -> dict:
    """Mini-batch CO DINH (theo umap_seed) dung de ve UMAP xuyen suot
    training, doc tu .npy CO SAN (khong tai lai)."""
    eval_tf = build_image_transforms(cfg, train=False)
    if dataset_name == "cifar100":
        base = NpyImageDataset(
            os.path.join(cfg.cifar100_dir, "train_images.npy"),
            os.path.join(cfg.cifar100_dir, "train_labels.npy"),
            transform=eval_tf)
    else:
        base = NpyImageDataset(
            os.path.join(cfg.cifar10_dir, "train_images.npy"),
            os.path.join(cfg.cifar10_dir, "train_labels.npy"),
            transform=eval_tf)

    rng = np.random.default_rng(cfg.umap_seed)
    indices = rng.choice(len(base), size=cfg.umap_probe_samples, replace=False).tolist()
    subset = _FixedIndexSubset(base, indices)
    loader = DataLoader(subset, batch_size=cfg.umap_probe_samples, shuffle=False)
    images, labels = next(iter(loader))
    return {"images": images, "labels": labels}


def build_landscape_loader(cfg: Config):
    """Loader nho, co dinh, dung de danh gia loss khi quet loss landscape
    (va khi uoc luong Hessian-trace sharpness), doc tu .npy CO SAN."""
    eval_tf = build_image_transforms(cfg, train=False)
    base = NpyImageDataset(
        os.path.join(cfg.cifar100_dir, "train_images.npy"),
        os.path.join(cfg.cifar100_dir, "train_labels.npy"),
        transform=eval_tf)
    n_needed = cfg.landscape_eval_batches * cfg.landscape_eval_batch_size
    rng = np.random.default_rng(cfg.landscape_seed)
    indices = rng.choice(len(base), size=n_needed, replace=False).tolist()
    subset = _FixedIndexSubset(base, indices)
    return DataLoader(subset, batch_size=cfg.landscape_eval_batch_size,
                      shuffle=False, num_workers=0, pin_memory=True, drop_last=True)


# =========================================================================
# MODEL: DETERMINISTIC ViT CLASSIFIER (KHONG BLL, KHONG posterior/particle)
# =========================================================================

class ViTClassifier(nn.Module):
    """
    ViT backbone (timm, deterministic) + 1 Linear classifier head thong
    thuong. Dung cho CA teacher va student trong standard Knowledge
    Distillation (Hinton et al., 2015) -- KHONG co thanh phan Bayesian
    nao (khac ban BLL truoc day: khong co posterior tren trong so, khong
    sample particle, khong ELBO/KL-to-prior).

    timm.create_model(..., num_classes=0) tra ve model voi `forward()` da
    cho ra dac trung sau pooling (CLS token / global pool), shape
    [B, embed_dim] (1024 cho vit_large_patch16_224, 384 cho
    vit_small_patch16_224, lay tu backbone.num_features), dung truc tiep
    lam input cho Linear head.
    """
    def __init__(self, backbone: nn.Module, head: nn.Linear):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    @classmethod
    def from_timm_name(cls, model_name, num_labels, pretrained):
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = backbone.num_features
        head = nn.Linear(embed_dim, num_labels)
        return cls(backbone, head)

    def enable_gradient_checkpointing(self):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)

    def backbone_features(self, images):
        return self.backbone(images)          # [B, D] (timm ViT, num_classes=0)

    def forward(self, images):
        h = self.backbone_features(images)
        return self.head(h)                   # [B, C] -- 1 vector logits / sample

    def landscape_named_parameters(self):
        """
        Tham so dung de xay random direction cho loss landscape. KHONG
        con gi can loai tru (khac ban BLL truoc day phai bo qua
        "log_sigma"): moi tham so o day deu deterministic.
        """
        yield from self.named_parameters()


# =========================================================================
# CHECKPOINTING
# =========================================================================

def save_checkpoint(model: ViTClassifier, cfg: Config, path: str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.backbone.state_dict(), os.path.join(path, "backbone.pt"))
    torch.save({
        "head_state_dict": model.head.state_dict(),
        "in_features":  model.head.in_features,
        "out_features": model.head.out_features,
    }, os.path.join(path, "head.pt"))
    with open(os.path.join(path, "marker.json"), "w") as f:
        json.dump({"is_kd_checkpoint": True}, f)
    print(f"[checkpoint] saved ViT classifier at: {path}")


def load_checkpoint(path: str, model_name: str) -> ViTClassifier:
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    backbone.load_state_dict(torch.load(os.path.join(path, "backbone.pt"), map_location="cpu"))
    payload = torch.load(os.path.join(path, "head.pt"), map_location="cpu")
    head = nn.Linear(payload["in_features"], payload["out_features"])
    head.load_state_dict(payload["head_state_dict"])
    return ViTClassifier(backbone, head)


def checkpoint_exists(path: str) -> bool:
    exists = os.path.isfile(os.path.join(path, "marker.json"))
    if exists:
        print(f"[checkpoint] Existing checkpoint found at: {path} -> skipping that stage.")
    return exists


# =========================================================================
# UMAP HELPERS
# =========================================================================

@torch.no_grad()
def collect_sample_logits(model: ViTClassifier, probe_batch: dict, cfg: Config
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Chay model tren probe_batch, thu logits [B, C] -- 1 diem / sample
    (khac ban BLL truoc day tra ve [B, K, C] voi K particle / sample)."""
    model.eval()
    images = probe_batch["images"].to(cfg.device)
    labels = probe_batch["labels"]

    with torch.autocast(
        device_type="cuda" if cfg.device == "cuda" else "cpu",
        dtype=cfg.mixed_precision_dtype,
    ):
        logits = model(images)      # [B, C]

    logits = logits.float().cpu()
    return logits.numpy(), labels.numpy()


def _run_umap(pts: np.ndarray, cfg: Config) -> np.ndarray:
    n_pts = pts.shape[0]
    n_neighbors = min(cfg.umap_n_neighbors, n_pts - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        n_epochs=cfg.umap_n_epochs,
        random_state=cfg.umap_seed,
        low_memory=False,
        verbose=False,
    )
    return reducer.fit_transform(pts)


def _label_color(label: int, num_labels: int):
    return _CMAP_100(label % 20 / 19.0) if num_labels > 2 else \
        (["#E05C5C", "#4C9BE8"][label])


def compute_sample_umap_data(
    pts: np.ndarray, labels: np.ndarray, tag: str, global_step: int,
    epoch: int, phase: str, cfg: Config,
) -> dict:
    """Chay UMAP va dong goi TOAN BO du lieu can de ve lai figure sau nay,
    tach rieng khoi phan ve matplotlib. 1 diem / sample (khong con
    "particle spread" nhu ban BLL)."""
    emb = _run_umap(pts, cfg)                # [B, 2]
    return {
        "embedding": emb.astype(np.float32),       # [B, 2]
        "labels": labels.astype(np.int64),          # [B]
        "tag": tag, "global_step": global_step, "epoch": epoch, "phase": phase,
        "n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist,
        "metric": cfg.umap_metric, "num_labels": cfg.num_labels,
    }


def plot_sample_umap(data: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """Ve scatter 1 panel tu du lieu da tinh san. Khong forward model /
    khong chay UMAP lai -- co the goi lai bao nhieu lan tuy y de doi style."""
    emb        = data["embedding"]
    labels     = data["labels"]
    tag        = data["tag"]
    num_labels = int(data["num_labels"])

    fig, ax = plt.subplots(figsize=style.figsize_single)
    colors = [_label_color(int(l), num_labels) for l in labels]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=style.marker_size * 3,
              alpha=0.75, linewidths=0, zorder=2)
    ax.set_title(
        f"[{tag.upper()}] UMAP of logits - step {data['global_step']}  "
        f"epoch {data['epoch']}  phase={data['phase']}\n"
        f"n_neighbors={data['n_neighbors']}  min_dist={data['min_dist']}  "
        f"metric={data['metric']}  |  {emb.shape[0]} samples",
        fontsize=style.subtitle_fontsize)
    ax.set_xlabel("UMAP dim 1", fontsize=style.label_fontsize)
    ax.set_ylabel("UMAP dim 2", fontsize=style.label_fontsize)
    ax.tick_params(labelsize=style.tick_fontsize)
    ax.grid(alpha=style.grid_alpha * 0.6)

    save_dir = os.path.join(cfg.figure_dir, "umap")
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{tag}_step_{int(data['global_step']):06d}.png"
    fpath = os.path.join(save_dir, fname)
    fig.tight_layout()
    fig.savefig(fpath, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[UMAP] saved -> {fpath}")


def run_and_plot_umap(model, probe_batch, tag, global_step, epoch, phase, cfg, style=DEFAULT_STYLE):
    """Tinh UMAP, luu du lieu tho, roi ve. Goi 1 lan duy nhat tai thoi diem train."""
    pts, lbs = collect_sample_logits(model, probe_batch, cfg)
    data = compute_sample_umap_data(
        pts, lbs, tag=tag, global_step=global_step, epoch=epoch, phase=phase, cfg=cfg,
    )
    save_figure_data(cfg, f"umap_{tag}_step_{global_step:06d}", data)
    plot_sample_umap(data, cfg, style)


def run_and_plot_dual_umap(teacher_model, student_model, teacher_probe, student_probe,
                           global_step, epoch, phase, cfg, style=DEFAULT_STYLE):
    run_and_plot_umap(teacher_model, teacher_probe, "teacher", global_step, epoch, phase, cfg, style)
    run_and_plot_umap(student_model, student_probe, "student", global_step, epoch, phase, cfg, style)


# =========================================================================
# STANDARD KNOWLEDGE DISTILLATION  (Hinton, Vinyals & Dean, 2015)
# =========================================================================

def hinton_kd_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                   cfg: Config) -> torch.Tensor:
    """
    Soft-target distillation loss kinh dien cua Hinton et al. (2015):

        L_soft = T^2 * KL( softmax(teacher_logits / T) || softmax(student_logits / T) )

    teacher_logits, student_logits: [B, C] -- 1 vector logits / sample
    (model deterministic, KHONG con nhieu particle nhu ban BLL truoc day).

    Nhan T^2 (T = cfg.kd_temperature) de can bang lai bien do gradient,
    dung theo de xuat goc cua Hinton et al. (2015): gradient cua soft-target
    loss ti le voi 1/T^2 so voi hard-target loss neu khong nhan lai.
    """
    T = cfg.kd_temperature
    teacher_log_probs = F.log_softmax(teacher_logits.float() / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)
    return kd


# =========================================================================
# SCHEDULE  (2-phase: task-only -> distillation ramp)
# =========================================================================

def student_schedule_weights(epoch, cfg: Config) -> float:
    """2-phase schedule: (1) task-only CE cho phase1_frac epoch dau,
    (2) + soft-target KD loss ramp tuyen tinh cho phan con lai. Tra ve
    1 gia tri duy nhat (alpha, trong so cua hinton_kd_loss()) -- khong con
    phase "posterior" (KL-to-prior) vi khong co BLL/ELBO."""
    n          = cfg.student_num_epochs
    phase1_end = max(1, round(n * cfg.phase1_frac))
    if epoch <= phase1_end:
        return 0.0
    frac = (epoch - phase1_end) / max(1, n - phase1_end)
    return cfg.kd_alpha_max * min(1.0, frac)


# =========================================================================
# EVALUATION (accuracy/F1/ECE in-distribution; predictive entropy for OOD)
# =========================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Naeini et al. 2015 / Guo et al. 2017).
    Chia [0,1] thanh n_bins bin deu theo confidence = max_c probs[n,c]; voi
    moi bin do |accuracy(bin) - confidence(bin)|, trong so theo |bin|/N.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_acc    = accuracies[in_bin].mean()
        bin_conf   = confidences[in_bin].mean()
        bin_weight = in_bin.sum() / n
        ece += bin_weight * abs(bin_acc - bin_conf)
    return float(ece)


@torch.no_grad()
def evaluate_classification_metrics(model: ViTClassifier, loader, device, dtype,
                                    ece_num_bins: int = 15):
    """1 forward deterministic / sample -> accuracy, f1 (macro), ECE.
    Khong con posterior-predictive / mean-forward / particle min-max nhu
    ban BLL truoc day, vi model chi co 1 output duy nhat cho moi input."""
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []
    total_loss, n_batches = 0.0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
            logits = model(images)

        logits = logits.float()
        ce = F.cross_entropy(logits, labels)
        probs = F.softmax(logits, dim=-1)

        all_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.detach().cpu().tolist())
        total_loss += ce.item()
        n_batches  += 1

    all_labels_np = np.array(all_labels, dtype=np.int64)
    all_probs_np  = np.concatenate(all_probs, axis=0)
    ece = compute_ece(all_probs_np, all_labels_np, n_bins=ece_num_bins)

    return {
        "loss":     total_loss / max(1, n_batches),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="macro"),
        "ece":      ece,
    }


@torch.no_grad()
def compute_predictive_entropy_scores(model: ViTClassifier, loader, cfg: Config,
                                      max_batches: Optional[int] = None) -> np.ndarray:
    """Entropy cua softmax deterministic cho tung sample -- dung lam OOD
    score. Entropy cao hon = bat dinh hon. Khong con trung binh qua nhieu
    particle nhu ban BLL truoc day (chi 1 forward / sample)."""
    model.eval()
    entropies = []
    for i, (images, _labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(cfg.device, non_blocking=True)
        with torch.autocast(device_type="cuda" if cfg.device == "cuda" else "cpu",
                            dtype=cfg.mixed_precision_dtype):
            logits = model(images).float()       # [B,C]
        probs = F.softmax(logits, dim=-1)          # [B,C]
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropies.extend(ent.cpu().tolist())
    return np.array(entropies, dtype=np.float64)


def evaluate_ood_auroc(model: ViTClassifier, id_loader, ood_loader, cfg: Config,
                       max_batches_each: int = 50) -> dict:
    """AUROC cho ID (CIFAR-100 test) vs OOD (CIFAR-10 test) dua tren
    predictive entropy. label 1 = OOD, score = entropy."""
    id_scores  = compute_predictive_entropy_scores(
        model, id_loader, cfg, max_batches=max_batches_each)
    ood_scores = compute_predictive_entropy_scores(
        model, ood_loader, cfg, max_batches=max_batches_each)
    y_true  = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    return {
        "auroc": float(auroc),
        "id_scores": id_scores, "ood_scores": ood_scores,
    }


# =========================================================================
# HESSIAN-TRACE SHARPNESS (Hutchinson estimator + double backprop, cung
# tinh than voi PyHessian / Yao et al. 2020). Hessian cua CE loss (1
# forward deterministic) tai theta* HIEN TAI (KHONG perturb).
# =========================================================================

def compute_hessian_trace(model: ViTClassifier, params: List[torch.Tensor],
                          images: torch.Tensor, labels: torch.Tensor,
                          num_hutchinson_samples: int, seed: Optional[int] = None) -> float:
    """
    Tr(H) ~= (1/M) * sum_i z_i^T H z_i, voi z_i ~ Rademacher i.i.d.
    (Hutchinson, 1990). Grad bac 1 (create_graph=True) chi tinh 1 LAN cho
    ca M mau; moi z_i chi can 1 lan backward bac 2 them. Generator rieng
    (khong dung torch.manual_seed toan cuc) de khong lam xao tron RNG
    stream chinh cua training loop.
    """
    logits = model(images)
    loss = F.cross_entropy(logits.float(), labels)
    first_grads = torch.autograd.grad(loss, params, create_graph=True)

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    trace_samples = []
    for i in range(num_hutchinson_samples):
        vecs = [
            (torch.randint(0, 2, p.shape, generator=gen).float() * 2.0 - 1.0).to(p.device, dtype=p.dtype)
            for p in params
        ]
        dot = sum((g * v).sum() for g, v in zip(first_grads, vecs))
        retain = i < num_hutchinson_samples - 1
        hv = torch.autograd.grad(dot, params, retain_graph=retain)
        trace_est = sum((h * v).sum().item() for h, v in zip(hv, vecs))
        trace_samples.append(trace_est)
    return float(np.mean(trace_samples))


def evaluate_hessian_trace_sharpness(model: ViTClassifier, loader, cfg: Config,
                                     seed: Optional[int] = None) -> float:
    """
    Uoc luong Hessian-trace sharpness cua model HIEN TAI, trung binh qua
    cfg.hessian_eval_batches batch co dinh (thuong la build_landscape_loader(cfg))
    va cfg.hessian_num_hutchinson_samples vector Rademacher moi batch.

    Can GRADIENT BAC 2 (double backprop) nen KHONG duoc goi duoi
    torch.no_grad(). Chay o FULL PRECISION (khong autocast) vi double
    backward voi bfloat16 de mat on dinh so hon backward bac 1. Dung
    torch.autograd.grad() (khong loss.backward()) nen an toan de goi xen
    giua training loop. LayerNorm khong co running stats nen khong can xu
    ly gi them o day (khac voi BatchNorm o ban CNN).
    """
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    batch_traces = []
    for i, (images, labels) in enumerate(loader):
        if i >= cfg.hessian_eval_batches:
            break
        images = images.to(cfg.device, non_blocking=True)
        labels = labels.to(cfg.device, non_blocking=True)
        batch_seed = None if seed is None else seed + i
        t = compute_hessian_trace(
            model, params, images, labels,
            num_hutchinson_samples=cfg.hessian_num_hutchinson_samples,
            seed=batch_seed,
        )
        batch_traces.append(t)

    if was_training:
        model.train()
    return float(np.mean(batch_traces)) if batch_traces else float("nan")


# =========================================================================
# STAGE 1: TEACHER FIT  (pretrained ViT-Large backbone + Linear head, CE)
# =========================================================================

def fit_teacher(cfg: Config) -> Tuple[str, List[Dict]]:
    """
    Teacher = ViT-Large PRETRAINED (timm, ImageNet weights) + Linear head
    moi khoi tao, huan luyen bang CE thong thuong (KHONG con ELBO/BLL).
    Backbone pretrained co the duoc finetune nhe (lr nho hon,
    cfg.teacher_backbone_lr_mult) hoac freeze hoan toan
    (teacher_finetune_backbone=False). So epoch fit teacher = so epoch
    distill student (CFG.teacher_num_epochs da duoc gan = CFG.student_num_epochs
    ngay sau khi tao CFG).
    """
    print(f"\n{'='*80}\nSTAGE 1: FITTING TEACHER (standard CE, pretrained backbone) -- {cfg.teacher_name}\n{'='*80}")

    model = ViTClassifier.from_timm_name(cfg.teacher_name, cfg.num_labels, cfg.teacher_pretrained)
    model.to(cfg.device)
    if cfg.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if not cfg.teacher_finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    print(f"[UMAP] Building teacher probe batch ({cfg.umap_probe_samples} samples)...")
    umap_probe = build_umap_probe_batch(cfg, dataset_name="cifar100")

    train_loader, eval_loader = build_cifar100_loaders(cfg, cfg.batch_size)

    if cfg.teacher_finetune_backbone:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": cfg.learning_rate * cfg.teacher_backbone_lr_mult},
            {"params": model.head.parameters(),     "lr": cfg.learning_rate},
        ]
    else:
        param_groups = [{"params": model.head.parameters(), "lr": cfg.learning_rate}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    total_steps  = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.teacher_num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps)) *
                     max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )

    history: List[Dict] = []
    model.train()
    global_step      = 0
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "teacher_vit_large_latest")

    epoch_pbar = tqdm(range(1, cfg.teacher_num_epochs + 1), desc="teacher epochs", unit="epoch")
    for epoch in epoch_pbar:
        epoch_ce, n_steps = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        step_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"teacher epoch {epoch}/{cfg.teacher_num_epochs}",
                         unit="step", leave=False)
        for step, (images, labels) in step_pbar:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            with torch.autocast(device_type="cuda" if cfg.device == "cuda" else "cpu",
                                dtype=cfg.mixed_precision_dtype):
                logits = model(images)
            logits = logits.float()
            ce = F.cross_entropy(logits, labels)
            loss = ce / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.umap_every_n_steps == 0:
                    run_and_plot_umap(model, umap_probe, "teacher", global_step, epoch, "fit", cfg)
                    model.train()

            epoch_ce += ce.item()
            n_steps  += 1
            step_pbar.set_postfix(ce=f"{ce.item():.4f}")

        avg_ce = epoch_ce / max(1, n_steps)
        metrics = evaluate_classification_metrics(
            model, eval_loader, cfg.device, cfg.mixed_precision_dtype,
            ece_num_bins=cfg.ece_num_bins)
        elapsed = time.time() - t0

        epoch_pbar.set_postfix(ce=f"{avg_ce:.4f}", test_acc=f"{metrics['accuracy']:.4f}")

        record = {
            "model": "teacher", "epoch": epoch, "ce_loss": avg_ce, "total_loss": avg_ce,
            "eval_loss": metrics["loss"], "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "ece": metrics["ece"], "epoch_time_sec": elapsed,
        }
        log_jsonl(cfg.log_file, record)
        history.append({
            "epoch": epoch, "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "total_loss": avg_ce, "ce_loss": avg_ce,
        })

        save_checkpoint(model, cfg, latest_ckpt_path)
        print(f"[latest teacher] epoch={epoch}  eval_loss={metrics['loss']:.4f}"
              f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  ece={metrics['ece']:.4f}")
        model.train()

    ckpt_path = os.path.join(cfg.checkpoint_dir, "teacher_vit_large")
    save_checkpoint(model, cfg, ckpt_path)

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path, history


# =========================================================================
# STAGE 2: STUDENT DISTILLATION  (ViT-Small, random init, standard Hinton KD)
# =========================================================================

def distill_student(cfg: Config, teacher_ckpt_path: str, teacher_model_for_umap: ViTClassifier
                    ) -> Tuple[str, List[Dict]]:
    """Student = ViT-Small KHONG pretrained (distill tu dau). 2-phase
    schedule (task-only -> KD ramp) + CE + standard Hinton (2015)
    soft-target KD loss. KHONG BLL/ELBO, KHONG Gromov-Wasserstein."""
    print(f"\n{'='*80}\nSTAGE 2: DISTILLING STUDENT (standard KD, Hinton 2015) -- {cfg.student_name}\n{'='*80}")

    student = ViTClassifier.from_timm_name(cfg.student_name, cfg.num_labels, cfg.student_pretrained)
    student.to(cfg.device)
    if cfg.use_gradient_checkpointing:
        student.enable_gradient_checkpointing()

    # Teacher dung de tinh logits distill (frozen, tu checkpoint).
    teacher = load_checkpoint(teacher_ckpt_path, cfg.teacher_name)
    teacher.to(cfg.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Teacher dung de ve UMAP (tach bien de giu dung kien truc goc).
    teacher_model_for_umap.to(cfg.device)
    teacher_model_for_umap.eval()
    for p in teacher_model_for_umap.parameters():
        p.requires_grad_(False)

    print(f"[UMAP] Building teacher probe batch ({cfg.umap_probe_samples} samples)...")
    teacher_umap_probe = build_umap_probe_batch(cfg, dataset_name="cifar100")
    print(f"[UMAP] Building student probe batch ({cfg.umap_probe_samples} samples)...")
    student_umap_probe = build_umap_probe_batch(cfg, dataset_name="cifar100")

    train_loader, eval_loader = build_cifar100_loaders(cfg, cfg.distill_batch_size)

    optimizer    = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps  = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.student_num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps)) *
                     max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )

    print(f"[student] batch={cfg.distill_batch_size}  epochs={cfg.student_num_epochs}  "
          f"T={cfg.kd_temperature}  alpha_max={cfg.kd_alpha_max}")

    history: List[Dict] = []
    student.train()
    global_step      = 0
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "student_vit_small_kd_latest")

    epoch_pbar = tqdm(range(1, cfg.student_num_epochs + 1), desc="student (KD) epochs", unit="epoch")
    for epoch in epoch_pbar:
        alpha = student_schedule_weights(epoch, cfg)
        phase = "task-only" if alpha == 0 else "distillation"
        print(f"\n[schedule] epoch {epoch}/{cfg.student_num_epochs}  phase={phase}  alpha={alpha:.3f}")

        epoch_ce, epoch_kd, n_steps = 0.0, 0.0, 0
        epoch_kd_term, epoch_total  = 0.0, 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        step_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"student epoch {epoch}/{cfg.student_num_epochs}",
                         unit="step", leave=False)
        for step, (images, labels) in step_pbar:
            images = images.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            with torch.no_grad(), torch.autocast(
                device_type="cuda" if cfg.device == "cuda" else "cpu",
                dtype=cfg.mixed_precision_dtype,
            ):
                teacher_logits = teacher(images).float()      # [B, C]

            with torch.autocast(device_type="cuda" if cfg.device == "cuda" else "cpu",
                                dtype=cfg.mixed_precision_dtype):
                student_logits = student(images)
            student_logits = student_logits.float()

            ce = F.cross_entropy(student_logits, labels)
            kd_loss = (hinton_kd_loss(teacher_logits, student_logits, cfg)
                      if alpha > 0.0 else torch.zeros((), device=cfg.device))

            kd_term   = alpha * kd_loss
            total_raw = ce + kd_term
            loss = total_raw / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.umap_every_n_steps == 0:
                    run_and_plot_dual_umap(
                        teacher_model=teacher_model_for_umap, student_model=student,
                        teacher_probe=teacher_umap_probe, student_probe=student_umap_probe,
                        global_step=global_step, epoch=epoch, phase=phase, cfg=cfg,
                    )
                    student.train()

            epoch_ce      += ce.item()
            epoch_kd      += float(kd_loss.item())
            epoch_kd_term += float(kd_term.item())
            epoch_total   += float(total_raw.item())
            n_steps       += 1

            step_pbar.set_postfix(ce=f"{ce.item():.4f}", kd=f"{float(kd_loss.item()):.4f}")

        avg_ce      = epoch_ce / max(1, n_steps)
        avg_kd      = epoch_kd / max(1, n_steps)
        avg_kd_term = epoch_kd_term / max(1, n_steps)
        avg_total   = epoch_total / max(1, n_steps)
        metrics = evaluate_classification_metrics(
            student, eval_loader, cfg.device, cfg.mixed_precision_dtype,
            ece_num_bins=cfg.ece_num_bins)
        elapsed = time.time() - t0

        epoch_pbar.set_postfix(ce=f"{avg_ce:.4f}", kd=f"{avg_kd:.4f}", test_acc=f"{metrics['accuracy']:.4f}")

        record = {
            "model": "student", "epoch": epoch, "phase": phase, "alpha": alpha,
            "ce_loss": avg_ce, "kd_loss": avg_kd, "kd_term": avg_kd_term, "total_loss": avg_total,
            "eval_loss": metrics["loss"], "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "ece": metrics["ece"], "epoch_time_sec": elapsed,
        }
        log_jsonl(cfg.log_file, record)
        history.append({
            "epoch": epoch, "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "total_loss": avg_total, "ce_loss": avg_ce, "kd_loss": avg_kd_term,
        })

        save_checkpoint(student, cfg, latest_ckpt_path)
        print(f"[latest student] epoch={epoch}  eval_loss={metrics['loss']:.4f}"
              f"  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  ece={metrics['ece']:.4f}")
        student.train()

    ckpt_path = os.path.join(cfg.checkpoint_dir, "student_vit_small_kd")
    save_checkpoint(student, cfg, ckpt_path)

    del student, teacher, teacher_model_for_umap, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path, history


# =========================================================================
# LOSS LANDSCAPE
# =========================================================================
# ViT dung LayerNorm (khong co running stats), nen KHONG can context-manager
# dac biet nhu o ban CNN (BatchNorm). LayerNorm affine (1 chieu) bi zero-out
# tu dong boi dieu kien "dim() <= 1" trong filter_normalize_direction(),
# giong het bias. Cung khong con tham so BLL nao can loai tru.

def get_random_direction_like(params):
    return [torch.randn_like(p) for p in params]


def filter_normalize_direction(direction, params):
    """
    Filter normalization (Li et al., 2018): voi moi tensor trong so p va
    huong ngau nhien d cung shape, scale d theo ti le ||p|| / ||d|| de
    "nhieu" co bien do tuong xung voi do lon cua chinh trong so do.

    Tensor 1 chieu (LayerNorm weight/bias, Linear bias, ...) khong co cau
    truc "filter" de chuan hoa theo nghia hinh hoc -> zero-out (giu nguyen
    tai theta*, khong di chuyen theo huong nay).
    """
    for d, p in zip(direction, params):
        if d.dim() <= 1:
            d.mul_(0.0)
            continue
        d_norm = d.norm()
        p_norm = p.norm()
        if d_norm.item() == 0:
            continue
        d.mul_(p_norm / (d_norm + 1e-10))


@torch.no_grad()
def apply_perturbation(params, base_params, dir1, dir2, alpha, beta):
    for p, p0, d1, d2 in zip(params, base_params, dir1, dir2):
        p.copy_(p0 + alpha * d1 + beta * d2)


@torch.no_grad()
def evaluate_classification_loss(model: ViTClassifier, loader, device, dtype, max_batches=None):
    """CE loss (1 forward deterministic) tren mot so batch co dinh, dung
    de quet loss landscape. Khong can xu ly gi dac biet cho LayerNorm
    (khac BatchNorm)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
            logits = model(images)
            loss   = F.cross_entropy(logits.float(), labels)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(20.0)
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(1, n_batches)


def compute_loss_landscape(model: ViTClassifier, loader, cfg: Config, seed=0):
    """Quet loss landscape tren luoi (alpha, beta) trong
    [cfg.landscape_alpha_range] x [cfg.landscape_beta_range]."""
    device = cfg.device
    grid   = cfg.landscape_grid_size
    torch.manual_seed(seed)
    named       = list(model.landscape_named_parameters())
    params      = [p for _, p in named]
    base_params = [p.detach().clone() for p in params]
    dir1 = get_random_direction_like(params)
    dir2 = get_random_direction_like(params)
    filter_normalize_direction(dir1, base_params)
    filter_normalize_direction(dir2, base_params)
    alphas    = np.linspace(cfg.landscape_alpha_range[0], cfg.landscape_alpha_range[1], grid)
    betas     = np.linspace(cfg.landscape_beta_range[0],  cfg.landscape_beta_range[1],  grid)
    loss_grid = np.zeros((grid, grid), dtype=np.float64)
    model.to(device)
    coords = [(i, a, j, b) for i, a in enumerate(alphas) for j, b in enumerate(betas)]
    pbar = tqdm(coords, total=grid*grid, desc="Loss landscape grid", unit="pt")

    for i, a, j, b in pbar:
        apply_perturbation(params, base_params, dir1, dir2, float(a), float(b))
        loss_grid[i, j] = evaluate_classification_loss(
            model, loader, device, cfg.mixed_precision_dtype,
            max_batches=cfg.landscape_eval_batches)
        pbar.set_postfix(alpha=f"{a:.2f}", beta=f"{b:.2f}", loss=f"{loss_grid[i,j]:.4f}")

    with torch.no_grad():
        for p, p0 in zip(params, base_params):
            p.copy_(p0)
    return alphas, betas, loss_grid


# ── Loss landscape: tach tinh toan / luu du lieu / ve, de "de ve lai" ────

def compute_landscape_data(teacher_model, student_model, cfg: Config, tag: str) -> dict:
    """Quet loss landscape cho ca teacher va student, tra ve dict de luu/ve."""
    landscape_loader = build_landscape_loader(cfg)
    results = {}
    for key, model in [(TEACHER_KEY, teacher_model), (STUDENT_KEY, student_model)]:
        model.to(cfg.device)
        print(f"\n[landscape:{tag}] Sweeping {cfg.landscape_grid_size}x{cfg.landscape_grid_size} for {key} ...")
        alphas, betas, loss_grid = compute_loss_landscape(
            model, landscape_loader, cfg, seed=cfg.landscape_seed)
        results[key] = {"alphas": alphas, "betas": betas, "loss_grid": loss_grid}
    return {"tag": tag, "results": results}


def save_landscape_data(cfg: Config, data: dict):
    tag = data["tag"]
    for key, r in data["results"].items():
        safe_key = key.split(" ")[0].replace("-", "_").lower()
        save_figure_data(cfg, f"landscape_{tag}_{safe_key}", {
            "alphas": r["alphas"].astype(np.float64),
            "betas": r["betas"].astype(np.float64),
            "loss_grid": r["loss_grid"].astype(np.float64),
            "model_key": key, "tag": tag,
        })


def _clip_for_display(lg: np.ndarray, low_pct: float = 1.0, high_pct: float = 90.0) -> np.ndarray:
    """Clip 2 phia theo percentile (mac dinh [1, 90]) de khong "nuot" chi
    tiet vung tam khi vai diem ngoai bien co bien do lon hon nhieu bac."""
    lo = np.percentile(lg, low_pct)
    hi = np.percentile(lg, high_pct)
    if hi <= lo:
        hi = lg.max()
    return np.clip(lg, lo, hi)


def _draw_3d_axis(ax, A, B, lg, key, fig, style: PlotStyle):
    """Mau va truc Z deu theo thang LINEAR (gia tri loss da clip)."""
    vmin = float(lg.min())
    vmax = float(lg.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    facecolors = plt.get_cmap(style.cmap_name)(norm(lg))
    surf = ax.plot_surface(A, B, lg, facecolors=facecolors, linewidth=0,
                           antialiased=True, edgecolor="none", shade=False)
    ax.set_title(f"{key} -- 3D (linear scale)", fontsize=style.subtitle_fontsize)
    ax.set_xlabel("alpha", fontsize=style.label_fontsize)
    ax.set_ylabel("beta", fontsize=style.label_fontsize)
    ax.set_zlabel("loss (clipped)", fontsize=style.label_fontsize)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=style.cmap_name)
    mappable.set_array(lg)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="loss")


def _draw_2d_axis(ax, A, B, lg, key, fig, style: PlotStyle):
    """Contour 2D voi thang mau LINEAR de phan giai vung day phang quanh theta*."""
    vmin = float(lg.min())
    vmax = float(lg.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 20)
    cs = ax.contourf(A, B, lg, levels=levels, cmap=style.cmap_name, norm=norm)
    ax.contour(A, B, lg, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    ax.set_title(f"{key} -- 2D (linear scale)", fontsize=style.subtitle_fontsize)
    ax.set_xlabel("alpha", fontsize=style.label_fontsize)
    ax.set_ylabel("beta", fontsize=style.label_fontsize)
    ax.scatter([0], [0], color="red", marker="*", s=150, label="theta* (trained weights)")
    ax.legend(loc="upper right", fontsize=style.legend_fontsize)
    fig.colorbar(cs, ax=ax, shrink=0.9, label="loss")


def plot_landscapes_all(data: dict, cfg: Config, title: str, save_name_prefix: str,
                        style: PlotStyle = DEFAULT_STYLE):
    """Ve tu du lieu landscape da tinh san (data tu compute_landscape_data hoac load_figure_data)."""
    results = data["results"]
    keys = list(results.keys())
    n    = len(keys)
    for suffix, rows_spec in [("_3d_1x2.png", [["3d"]]), ("_2d_1x2.png", [["2d"]]),
                              ("_2x2.png", [["3d"], ["2d"]])]:
        fig = plt.figure(figsize=(style.figsize_single[0]*n, style.figsize_single[1]*len(rows_spec)))
        for r, row in enumerate(rows_spec):
            mode = row[0]
            for c, key in enumerate(keys):
                rdata = results[key]
                alphas, betas, lg = rdata["alphas"], rdata["betas"], rdata["loss_grid"]
                A, B = np.meshgrid(alphas, betas, indexing="ij")
                lgc  = _clip_for_display(lg)
                idx  = r * n + c + 1
                if mode == "3d":
                    ax = fig.add_subplot(len(rows_spec), n, idx, projection="3d")
                    _draw_3d_axis(ax, A, B, lgc, key, fig, style)
                else:
                    ax = fig.add_subplot(len(rows_spec), n, idx)
                    _draw_2d_axis(ax, A, B, lgc, key, fig, style)
        fig.suptitle(title, fontsize=style.title_fontsize, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(cfg.figure_dir, save_name_prefix + suffix)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=style.dpi)
        plt.close(fig)
        print(f"[saved figure] {p}")


def run_landscape_and_plot(teacher_model, student_model, cfg: Config, tag: str,
                           title: str, save_name_prefix: str, style: PlotStyle = DEFAULT_STYLE):
    data = compute_landscape_data(teacher_model, student_model, cfg, tag)
    save_landscape_data(cfg, data)
    plot_landscapes_all(data, cfg, title=title, save_name_prefix=save_name_prefix, style=style)
    return data


# =========================================================================
# METRIC / LOSS CURVES
# =========================================================================

def save_history_data(cfg: Config, teacher_history: List[Dict], student_history: List[Dict]):
    payload = {"teacher_history": teacher_history, "student_history": student_history,
              "teacher_key": TEACHER_KEY, "student_key": STUDENT_KEY}
    os.makedirs(cfg.figure_data_dir, exist_ok=True)
    with open(os.path.join(cfg.figure_data_dir, "training_history.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_history_data(cfg: Config) -> dict:
    with open(os.path.join(cfg.figure_data_dir, "training_history.json")) as f:
        return json.load(f)


def plot_metric_curves(history: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """Accuracy/F1 vs epoch cho teacher va student. Duong don gian (khong
    con band [min, max] giua cac particle nhu ban BLL, vi model o day
    deterministic -- chi co 1 gia tri accuracy/f1 duy nhat / epoch)."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_wide)

    for key, label, color in [
        (history.get("teacher_key", TEACHER_KEY), "ViT-Large teacher", style.teacher_color),
        (history.get("student_key", STUDENT_KEY), "ViT-Small student (KD)", style.student_color),
    ]:
        rows = history.get("teacher_history" if "teacher" in key.lower() else "student_history", [])
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        acc    = [r["accuracy"] for r in rows]
        f1     = [r["f1"] for r in rows]
        axes[0].plot(epochs, acc, marker="o", linewidth=style.line_width,
                    markersize=style.marker_size, label=label, color=color)
        axes[1].plot(epochs, f1, marker="o", linewidth=style.line_width,
                    markersize=style.marker_size, label=label, color=color)

    for ax, title, ylabel in [
        (axes[0], "Test Accuracy vs Epoch (CIFAR-100)", "Accuracy"),
        (axes[1], "Test F1 (macro) vs Epoch (CIFAR-100)", "F1 Score"),
    ]:
        ax.set_title(title, fontsize=style.subtitle_fontsize)
        ax.set_xlabel("Epoch", fontsize=style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.grid(alpha=style.grid_alpha)
        ax.legend(fontsize=style.legend_fontsize, loc="lower right")
    fig.suptitle("CIFAR-100 Classification: Teacher (ViT-Large) vs Student (ViT-Small, standard Hinton KD)",
                fontsize=style.title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(cfg.figure_dir, "accuracy_f1_vs_epoch.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=style.dpi)
    plt.close(fig)
    print(f"[saved figure] {p}")


def plot_loss_curves(history: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """3 panel: total_loss / ce_loss / kd_loss (thay cho 4 panel truoc day
    total/ce/kl/gw -- bo panel kl_loss vi khong con ELBO)."""
    panels = [
        ("total_loss", "Total Loss vs Epoch", "Total loss"),
        ("ce_loss",    "Cross-Entropy Loss vs Epoch", "CE loss"),
        ("kd_loss",    "Soft-Target KD Loss (weighted) vs Epoch", "alpha . KD (Hinton 2015)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(style.figsize_wide[0] * 1.25, style.figsize_wide[1]))

    teacher_rows = history.get("teacher_history", [])
    student_rows = history.get("student_history", [])

    for ax, (field, title, ylabel) in zip(axes, panels):
        plotted_any = False
        for label, color, rows in [
            ("ViT-Large teacher", style.teacher_color, teacher_rows),
            ("ViT-Small student (KD)", style.student_color, student_rows),
        ]:
            sub = [r for r in rows if field in r]
            if not sub:
                continue
            epochs = [r["epoch"] for r in sub]
            vals   = [r[field]   for r in sub]
            ax.plot(epochs, vals, marker="o", markersize=style.marker_size,
                    linewidth=style.line_width, label=label, color=color)
            plotted_any = True
        ax.set_title(title, fontsize=style.subtitle_fontsize)
        ax.set_xlabel("Epoch", fontsize=style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.grid(alpha=style.grid_alpha)
        if plotted_any:
            ax.legend(fontsize=style.legend_fontsize)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=style.label_fontsize, color="gray")

    fig.suptitle("Loss Components vs Epoch: Teacher (ViT-Large) vs Student (ViT-Small, standard KD)",
                fontsize=style.title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(cfg.figure_dir, "loss_curves.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=style.dpi)
    plt.close(fig)
    print(f"[saved figure] {p}")


# =========================================================================
# OOD PLOT  (CIFAR-100 ID vs CIFAR-10 OOD, predictive entropy + AUROC)
# =========================================================================

def run_and_save_ood_eval(teacher_model, student_model, cfg: Config) -> dict:
    """Tinh AUROC OOD (CIFAR-10) cho ca teacher va student, luu du lieu tho."""
    id_loader  = build_cifar100_loaders(cfg, cfg.batch_size)[1]    # eval split
    ood_loader = build_cifar10_ood_loader(cfg, cfg.batch_size)

    results = {}
    for key, model in [(TEACHER_KEY, teacher_model), (STUDENT_KEY, student_model)]:
        model.to(cfg.device)
        print(f"\n[OOD] Evaluating {key} on ID=CIFAR-100 test vs OOD=CIFAR-10 test ...")
        r = evaluate_ood_auroc(model, id_loader, ood_loader, cfg)
        results[key] = r
        print(f"[OOD] {key}: AUROC = {r['auroc']:.4f}")

    payload = {}
    for key, r in results.items():
        safe_key = key.split(" ")[0].replace("-", "_").lower()
        payload[f"{safe_key}_id_scores"]  = np.array(r["id_scores"])
        payload[f"{safe_key}_ood_scores"] = np.array(r["ood_scores"])
        payload[f"{safe_key}_auroc"]      = r["auroc"]
        payload[f"{safe_key}_model_key"]  = key
    save_figure_data(cfg, "ood_eval", payload)
    return {key: {"auroc": r["auroc"]} for key, r in results.items()}, payload


def plot_ood_eval(payload: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """Histogram predictive-entropy ID vs OOD cho teacher va student (2
    subplot), kem AUROC trong title."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_wide)
    for ax, prefix, label, color in [
        (axes[0], "vit_large", "ViT-Large teacher", style.teacher_color),
        (axes[1], "vit_small", "ViT-Small student (KD)", style.student_color),
    ]:
        id_scores  = payload.get(f"{prefix}_id_scores")
        ood_scores = payload.get(f"{prefix}_ood_scores")
        auroc      = payload.get(f"{prefix}_auroc")
        if id_scores is None or ood_scores is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.hist(id_scores, bins=30, alpha=0.6, density=True, color="tab:green",
               label="ID: CIFAR-100 test")
        ax.hist(ood_scores, bins=30, alpha=0.6, density=True, color="tab:red",
               label="OOD: CIFAR-10 test")
        auroc_str = f"{auroc:.4f}" if auroc is not None else "n/a"
        ax.set_title(f"{label}\nAUROC = {auroc_str}", fontsize=style.subtitle_fontsize)
        ax.set_xlabel("Predictive entropy (softmax)", fontsize=style.label_fontsize)
        ax.set_ylabel("Density", fontsize=style.label_fontsize)
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.legend(fontsize=style.legend_fontsize)
        ax.grid(alpha=style.grid_alpha)

    fig.suptitle("OOD Detection: train/ID = CIFAR-100, OOD = CIFAR-10",
                fontsize=style.title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(cfg.figure_dir, "ood_entropy_auroc.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=style.dpi)
    plt.close(fig)
    print(f"[saved figure] {p}")


# =========================================================================
# RE-PLOT HELPER  (ve lai TAT CA figure tu du lieu da luu, KHONG can GPU)
# =========================================================================

def replot_all_from_saved(cfg: Config = CFG, style: PlotStyle = DEFAULT_STYLE):
    """Doc lai toan bo du lieu da luu trong cfg.figure_data_dir va ve lai
    cac figure chinh voi mot PlotStyle moi, khong can chay lai training."""
    hist = load_history_data(cfg)
    plot_metric_curves(hist, cfg, style)
    plot_loss_curves(hist, cfg, style)

    for tag, title, prefix in [
        ("BEFORE", "Loss Landscape BEFORE Knowledge Distillation", "loss_landscape_BEFORE_distillation"),
        ("AFTER",  "Loss Landscape AFTER Knowledge Distillation",  "loss_landscape_AFTER_distillation"),
    ]:
        results = {}
        for key in [TEACHER_KEY, STUDENT_KEY]:
            safe_key = key.split(" ")[0].replace("-", "_").lower()
            try:
                d = load_figure_data(cfg, f"landscape_{tag}_{safe_key}")
                results[key] = {"alphas": d["alphas"], "betas": d["betas"], "loss_grid": d["loss_grid"]}
            except FileNotFoundError:
                continue
        if results:
            plot_landscapes_all({"results": results}, cfg, title=title, save_name_prefix=prefix, style=style)

    try:
        ood_payload = load_figure_data(cfg, "ood_eval")
        plot_ood_eval(ood_payload, cfg, style)
    except FileNotFoundError:
        pass

    print("[replot] done.")


# =========================================================================
# MAIN
# =========================================================================

def main():
    os.makedirs(CFG.output_dir,       exist_ok=True)
    os.makedirs(CFG.checkpoint_dir,   exist_ok=True)
    os.makedirs(CFG.figure_dir,       exist_ok=True)
    os.makedirs(CFG.figure_data_dir,  exist_ok=True)
    set_seed(CFG.seed)
    print(f"Device: {CFG.device}")
    if CFG.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Teacher epochs == Student epochs (yeu cau): "
          f"{CFG.teacher_num_epochs} == {CFG.student_num_epochs}")

    teacher_ckpt_path = os.path.join(CFG.checkpoint_dir, "teacher_vit_large")
    teacher_history: List[Dict] = []

    if checkpoint_exists(teacher_ckpt_path):
        print(">>> Teacher checkpoint found. Skipping teacher fit.")
    else:
        if CFG.run_pre_landscape:
            print("\n>>> STEP 1: Loss landscape BEFORE distillation (random/pretrained init) <<<")
            t_fresh = ViTClassifier.from_timm_name(CFG.teacher_name, CFG.num_labels, CFG.teacher_pretrained)
            s_fresh = ViTClassifier.from_timm_name(CFG.student_name, CFG.num_labels, CFG.student_pretrained)
            run_landscape_and_plot(
                t_fresh, s_fresh, CFG, tag="BEFORE",
                title="Loss Landscape BEFORE Knowledge Distillation",
                save_name_prefix="loss_landscape_BEFORE_distillation",
            )
            del t_fresh, s_fresh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if CFG.run_teacher_fit:
            print(f"\n>>> STEP 2: Fitting teacher on pretrained ViT-Large "
                  f"({CFG.teacher_num_epochs} epochs, = student_num_epochs) <<<")
            teacher_ckpt_path, teacher_history = fit_teacher(CFG)

    student_ckpt_path = os.path.join(CFG.checkpoint_dir, "student_vit_small_kd")
    student_history: List[Dict] = []

    if CFG.run_distillation:
        print(f"\n>>> STEP 3: Distilling student (standard Hinton KD, "
              f"{CFG.student_num_epochs} epochs) <<<")
        print(f"[distill] Loading teacher checkpoint for UMAP: {teacher_ckpt_path}")
        teacher_for_umap = load_checkpoint(teacher_ckpt_path, CFG.teacher_name)
        teacher_for_umap.eval()
        for p in teacher_for_umap.parameters():
            p.requires_grad_(False)

        student_ckpt_path, student_history = distill_student(
            CFG, teacher_ckpt_path=teacher_ckpt_path, teacher_model_for_umap=teacher_for_umap,
        )

    if teacher_history or student_history:
        save_history_data(CFG, teacher_history, student_history)

    if CFG.run_post_landscape:
        print("\n>>> STEP 4: Loss landscape AFTER distillation <<<")
        teacher_model = load_checkpoint(teacher_ckpt_path, CFG.teacher_name)
        student_model = load_checkpoint(student_ckpt_path, CFG.student_name)
        run_landscape_and_plot(
            teacher_model, student_model, CFG, tag="AFTER",
            title="Loss Landscape AFTER Knowledge Distillation",
            save_name_prefix="loss_landscape_AFTER_distillation",
        )

        if CFG.run_ood_eval:
            print("\n>>> STEP 4b: OOD evaluation (ID=CIFAR-100, OOD=CIFAR-10) <<<")
            _, ood_payload = run_and_save_ood_eval(teacher_model, student_model, CFG)
            plot_ood_eval(ood_payload, CFG)

        del teacher_model, student_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if CFG.run_metric_curves and (teacher_history or student_history):
        print("\n>>> STEP 5: Plotting accuracy / F1 curves <<<")
        hist_payload = {"teacher_history": teacher_history, "student_history": student_history,
                        "teacher_key": TEACHER_KEY, "student_key": STUDENT_KEY}
        plot_metric_curves(hist_payload, CFG)

        print("\n>>> STEP 6: Plotting loss curves (total / CE / KD) <<<")
        plot_loss_curves(hist_payload, CFG)

    print("\nFull pipeline complete.")
    print(f"Figures      : {CFG.figure_dir}")
    print(f"Figure data  : {CFG.figure_data_dir}  (dung de ve lai, xem replot_all_from_saved())")
    print(f"Checkpoints  : {CFG.checkpoint_dir}")
    print(f"Log          : {CFG.log_file}")


if __name__ == "__main__":
    main()