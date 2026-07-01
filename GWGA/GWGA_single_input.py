"""
GWGA_single_input_vit.py  (ViT setup)
=====================================
Gromov-Wasserstein Alignment of Bayesian Last-Layer Posteriors
(Structural Knowledge Distillation) -- SINGLE-INPUT variant (Algorithm 1
trong paper, KHONG phai Algorithm 2 batch-pooled).

Kien truc & du lieu -- Vision Transformer, gan voi tinh than setup "CV-1
(large-scale classification): Large -> small vision transformer" trong
paper Sec 6.2, nhung GIU LAI bo metric cua CV-2 (accuracy, ECE,
Hessian-trace sharpness) vi van muon do necessary-condition gap cua
Proposition 5.3:
  - Teacher  : ViT-Large/16 (timm, pretrained ImageNet) + Bayesian Last
               Layer (BLL). Backbone pretrained co the duoc finetune nhe
               hoac freeze; chi BLL head la thanh phan Bayesian moi.
               So epoch fit teacher = so epoch distill student.
  - Student  : ViT-Small/16 (timm, KHONG pretrained -- random init) + BLL,
               distill tu dau bang ELBO + GW structural loss.
  - Train set       : CIFAR-100 (num_labels = 100), anh duoc UPSIZE tu
                       32x32 len 224x224 (img_size) de khop input cua
                       ViT-Large/Small patch16/224.
  - Eval (in-dist)  : CIFAR-100 test split.
  - OOD test        : CIFAR-10 (AUROC / OOD detection, KHONG dung de train).

Single-input GW (Algorithm 1): voi MOI sample trong batch, sample K
particles tu posterior cua teacher/student tai INPUT DO, dung K dap so do
de xay cost matrix C in R^{KxK} (pairwise distance giua K particles CUA
CUNG MOT INPUT). GW duoc giai DOC LAP cho tung sample, vector hoa qua
batch dim B (khac Algorithm 2 batch-pooled, gop (nK)x(nK)).

So voi ban CNN/ResNet truoc -- diem khac biet quan trong:
  ViT (timm) dung LayerNorm, KHONG dung BatchNorm. LayerNorm:
    - KHONG co running statistics (buffer) nhu BatchNorm -- thong ke duoc
      tinh TUC THOI tren chinh moi forward pass, bat ke dang train hay
      eval. Vi vay loss landscape KHONG can bat ky context-manager dac
      biet nao de "tam thoi chuyen sang dung batch statistics" nhu ban
      CNN (xem _bn_use_batch_stats_context cu) -- van de do KHONG TON TAI
      voi LayerNorm.
    - Affine weight/bias cua LayerNorm la vector 1 chieu (1 gia tri/dim),
      giong Conv/Linear bias -- da duoc dieu kien "d.dim() <= 1" trong
      filter_normalize_direction() zero-out tu dong (xem Li et al., 2018),
      nen KHONG can logic loai-tru rieng nhu BN affine o ban CNN.
  -> Ket qua: landscape_named_parameters() va compute_loss_landscape()
     don gian hon han ban CNN (khong can quet/khoi-phuc trang thai BN).

Cac diem giu nguyen (logic khong doi, chi doi kien truc/domain):
  - 1 Config dataclass duy nhat.
  - Loss landscape duoc ve TRUOC khi train (random init ca 2 model) va
    SAU khi distill xong (filter-normalized random directions, 1x2/2x2 grid).
  - UMAP duoc ve moi `umap_every_n_steps` cho CA teacher lan student trong
    luc distill (cung 1 probe batch co dinh xuyen suot training).
  - 3-phase schedule (task-only -> posterior -> structural) cho student.
  - ECE, Hessian-trace sharpness (Hutchinson estimator), OOD AUROC
    (CIFAR-100 ID vs CIFAR-10 OOD) deu giu nguyen.
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
#   GWGA-CV1-VIT/
#     dataset/
#       cifar-10/   {train,test}_images.npy, {train,test}_labels.npy, classes.txt
#       cifar-100/  {train,test}_images.npy, {train,test}_labels.npy,
#                   {train,test}_coarse_labels.npy, fine_classes.txt, coarse_classes.txt
#     GWGA/
#       GWGA_single_input_vit.py   <-- file nay
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset")
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

    # ── Teacher fit (tren pretrained backbone) ───────────────────────────
    # YEU CAU: so epoch fit teacher = so epoch distill student -> gan o
    # ngay sau khi tao CFG (xem duoi class Config).
    teacher_num_epochs: int = 10
    teacher_kl_beta_max: float = 1.0
    teacher_kl_warmup_frac: float = 0.1
    teacher_finetune_backbone: bool = True
    teacher_backbone_lr_mult: float = 0.1   # backbone lr = learning_rate * mult

    # ── Student distillation ─────────────────────────────────────────────
    student_num_epochs: int = 30
    phase1_frac: float = 0.3
    phase2_frac: float = 0.3
    phase3_frac: float = 0.4
    kl_beta_max: float = 1.0
    gw_gamma_max: float = 1.0

    # ── Optimization ──────────────────────────────────────────────────────
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    mixed_precision_dtype: torch.dtype = torch.bfloat16

    batch_size: int = 512
    distill_batch_size: int = 256
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True   # timm ViT ho tro set_grad_checkpointing()

    # ── Bayesian Last Layer ──────────────────────────────────────────────
    num_particles: int = 16
    eval_num_particles: int = 16
    prior_std: float = 1.0
    init_log_sigma: float = -2.3

    # ── Gromov-Wasserstein (single-input, K x K cost matrix moi sample) ──
    gw_epsilon: float = 0.1
    gw_sinkhorn_iters: int = 30
    gw_outer_iters: int = 10
    gw_distance: str = "cosine"     # "sqeuclidean" | "cosine"

    # ── UMAP probe ────────────────────────────────────────────────────────
    umap_every_n_steps: int = 100
    umap_probe_samples: int = 256
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_n_epochs: int = 500
    umap_seed: int = 42

    # ── Loss landscape ───────────────────────────────────────────────────
    landscape_grid_size: int = 15
    landscape_alpha_range: Tuple[float, float] = (-1.0, 1.0)
    landscape_beta_range: Tuple[float, float] = (-1.0, 1.0)
    landscape_eval_batches: int = 10
    landscape_eval_batch_size: int = 16
    landscape_seed: int = 42
    # Khong can co flag/context-manager rieng cho LayerNorm nhu BatchNorm
    # o ban CNN -- LayerNorm khong co running stats nen khong co van de
    # "thong ke cu khong khop voi trong so moi" khi quet (alpha, beta).

    # ── ECE (Expected Calibration Error) ─────────────────────────────────
    ece_num_bins: int = 15

    # ── Hessian-trace sharpness (necessary-condition gap, Proposition 5.3) ─
    hessian_num_hutchinson_samples: int = 10
    hessian_eval_batches: int = 5
    hessian_seed: int = 777

    # ── OOD evaluation (CIFAR-10 dung lam OOD cho model train tren CIFAR-100) ─
    ood_num_particles: int = 16
    ood_score_type: str = "predictive_entropy"

    current_dir = Path(__file__).resolve().parent
    base_output_dir = current_dir / "output_vit_cifar_bll_gw_single_input"

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

TEACHER_KEY = "ViT-Large (teacher, BLL)"
STUDENT_KEY = "ViT-Small (student, BLL+GW)"

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
# SIGMA HELPER
# =========================================================================

def get_sigma_stats(model) -> dict:
    with torch.no_grad():
        sigma = model.head.log_sigma.exp()
        return {
            "sigma_mean": sigma.mean().item(),
            "sigma_min":  sigma.min().item(),
            "sigma_max":  sigma.max().item(),
        }


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
# BAYESIAN LAST LAYER
# =========================================================================

class BayesianLastLayer(nn.Module):
    """Mean-field Gaussian posterior tren last layer: q(W) = N(mu, diag(sigma^2)).
    Tuong ung Eq. (3) trong paper."""
    def __init__(self, in_features, out_features, prior_std=1.0, init_log_sigma=-2.3):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.prior_std    = prior_std
        self.mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.mu, mean=0.0, std=1.0 / math.sqrt(in_features))
        self.log_sigma = nn.Parameter(
            torch.full((out_features, in_features), float(init_log_sigma))
        )

    def sample_weights(self, num_particles):
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn(num_particles, self.out_features, self.in_features,
                          device=self.mu.device, dtype=self.mu.dtype)
        return self.mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    def forward(self, h, num_particles):
        """h: [B, D] -> logits: [B, K, C]."""
        W = self.sample_weights(num_particles)        # [K, out, in]
        return torch.einsum("bd,kcd->bkc", h, W)      # [B, K, out]

    def forward_mean(self, h):
        return F.linear(h, self.mu)

    def kl_divergence(self):
        sigma2    = torch.exp(2.0 * self.log_sigma)
        prior_var = self.prior_std ** 2
        kl = 0.5 * ((sigma2 + self.mu ** 2) / prior_var - 1.0 - torch.log(sigma2 / prior_var))
        return kl.sum()


class BLLViTClassifier(nn.Module):
    """
    ViT backbone (timm, deterministic) + Bayesian Last Layer head.
    timm.create_model(..., num_classes=0) tra ve model voi `forward()` da
    cho ra dac trung sau pooling (CLS token / global pool), shape
    [B, embed_dim] (1024 cho vit_large_patch16_224, 384 cho
    vit_small_patch16_224, lay tu backbone.num_features), dung truc tiep
    lam input h cho BLL head.
    """
    def __init__(self, backbone: nn.Module, head: BayesianLastLayer):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    @classmethod
    def from_timm_name(cls, model_name, num_labels, pretrained, prior_std, init_log_sigma):
        backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = backbone.num_features
        head = BayesianLastLayer(embed_dim, num_labels,
                                 prior_std=prior_std, init_log_sigma=init_log_sigma)
        return cls(backbone, head)

    def enable_gradient_checkpointing(self):
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable=True)

    def backbone_features(self, images):
        return self.backbone(images)         # [B, D] (timm ViT, num_classes=0)

    def forward(self, images, num_particles):
        h = self.backbone_features(images)
        return self.head(h, num_particles)   # [B, K, C]

    def forward_mean(self, images):
        h = self.backbone_features(images)
        return self.head.forward_mean(h)     # [B, C]

    def kl_divergence(self):
        return self.head.kl_divergence()

    def landscape_named_parameters(self):
        """
        Tham so dung de xay random direction cho loss landscape. Chi loai
        tru "log_sigma" cua BLL head (khong phai thanh phan deterministic
        cua loss landscape theo nghia thong thuong).

        KHAC voi ban CNN (ResNet + BatchNorm): ViT dung LayerNorm, vector
        affine weight/bias cua no la 1 chieu va se TU DONG bi zero-out boi
        dieu kien "d.dim() <= 1" trong filter_normalize_direction() --
        giong het cach Conv/Linear bias duoc xu ly -- nen KHONG can logic
        loai-tru rieng (id-based) nhu BatchNorm affine o ban CNN.
        """
        for name, p in self.named_parameters():
            if "log_sigma" in name:
                continue
            yield name, p


# =========================================================================
# CHECKPOINTING
# =========================================================================

def save_bll_checkpoint(model: BLLViTClassifier, cfg: Config, path: str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.backbone.state_dict(), os.path.join(path, "backbone.pt"))
    torch.save({
        "head_state_dict": model.head.state_dict(),
        "in_features":  model.head.in_features,
        "out_features": model.head.out_features,
        "prior_std":    model.head.prior_std,
    }, os.path.join(path, "bll_head.pt"))
    with open(os.path.join(path, "bll_marker.json"), "w") as f:
        json.dump({"is_bll_checkpoint": True}, f)
    print(f"[checkpoint] saved BLL ViT model at: {path}")


def load_bll_checkpoint(path: str, model_name: str) -> BLLViTClassifier:
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
    backbone.load_state_dict(torch.load(os.path.join(path, "backbone.pt"), map_location="cpu"))
    payload = torch.load(os.path.join(path, "bll_head.pt"), map_location="cpu")
    head = BayesianLastLayer(payload["in_features"], payload["out_features"],
                             prior_std=payload["prior_std"])
    head.load_state_dict(payload["head_state_dict"])
    return BLLViTClassifier(backbone, head)


def bll_checkpoint_exists(path: str) -> bool:
    exists = os.path.isfile(os.path.join(path, "bll_marker.json"))
    if exists:
        print(f"[checkpoint] Existing BLL checkpoint found at: {path} -> skipping that stage.")
    return exists


# =========================================================================
# UMAP HELPERS
# =========================================================================

@torch.no_grad()
def collect_particle_logits(model: BLLViTClassifier, probe_batch: dict, cfg: Config
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Chay model tren probe_batch voi num_particles, thu logits [B, K, C].
    Tra ve pts [B*K, C] va labels [B*K] (label sample goc lap lai K lan)."""
    model.eval()
    images = probe_batch["images"].to(cfg.device)
    labels = probe_batch["labels"]

    with torch.autocast(
        device_type="cuda" if cfg.device == "cuda" else "cpu",
        dtype=cfg.mixed_precision_dtype,
    ):
        logits = model(images, cfg.num_particles)      # [B, K, C]

    logits = logits.float().cpu()
    B, K, C = logits.shape
    pts = logits.reshape(B * K, C).numpy()
    lbs = labels.unsqueeze(1).expand(B, K).reshape(B * K).numpy()
    return pts, lbs


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


def compute_particle_umap_data(
    pts: np.ndarray, labels: np.ndarray, tag: str, global_step: int,
    epoch: int, phase: str, sigma_mean: float, cfg: Config, num_particles: int,
) -> dict:
    """Chay UMAP va dong goi TOAN BO du lieu can de ve lai figure sau nay,
    tach rieng khoi phan ve matplotlib."""
    emb = _run_umap(pts, cfg)                # [B*K, 2]
    B = pts.shape[0] // num_particles
    return {
        "embedding": emb.astype(np.float32),       # [B*K, 2]
        "labels": labels.astype(np.int64),          # [B*K]
        "tag": tag, "global_step": global_step, "epoch": epoch, "phase": phase,
        "sigma_mean": sigma_mean, "num_particles": num_particles, "B": B,
        "n_neighbors": cfg.umap_n_neighbors, "min_dist": cfg.umap_min_dist,
        "metric": cfg.umap_metric, "num_labels": cfg.num_labels,
    }


def plot_particle_umap(data: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """Ve scatter 2 panel tu du lieu da tinh san. Khong forward model /
    khong chay UMAP lai -- co the goi lai bao nhieu lan tuy y de doi style."""
    emb            = data["embedding"]
    labels         = data["labels"]
    tag            = data["tag"]
    num_particles  = int(data["num_particles"])
    B              = int(data["B"])
    num_labels     = int(data["num_labels"])

    fig, axes = plt.subplots(1, 2, figsize=(style.figsize_wide[0] + 2, style.figsize_wide[1] + 1.5))

    ax = axes[0]
    for k in range(num_particles):
        idx    = np.arange(B) * num_particles + k
        x, y   = emb[idx, 0], emb[idx, 1]
        cols   = [_label_color(int(labels[i]), num_labels) for i in idx]
        size   = style.marker_size * 11 if k == 0 else style.marker_size * 2.5
        alpha  = 0.85 if k == 0 else 0.30
        marker = "*" if k == 0 else "o"
        ax.scatter(x, y, c=cols, s=size, alpha=alpha, marker=marker,
                   linewidths=0, zorder=3 if k == 0 else 2)
    ax.scatter([], [], c="gray", s=style.marker_size * 11, marker="*",
               label="Particle 0 (anchor)", alpha=0.9)
    ax.scatter([], [], c="gray", s=style.marker_size * 2.5, marker="o",
               label=f"Particles 1-{num_particles-1}", alpha=0.4)
    ax.legend(loc="upper right", fontsize=style.legend_fontsize, framealpha=0.8)
    ax.set_title(f"All {num_particles} particles (B*K = {B*num_particles} pts)",
                fontsize=style.subtitle_fontsize)
    ax.set_xlabel("UMAP dim 1", fontsize=style.label_fontsize)
    ax.set_ylabel("UMAP dim 2", fontsize=style.label_fontsize)
    ax.tick_params(labelsize=style.tick_fontsize)
    ax.grid(alpha=style.grid_alpha * 0.6)

    ax2 = axes[1]
    for b in range(B):
        idx   = b * num_particles + np.arange(num_particles)
        xs, ys = emb[idx, 0], emb[idx, 1]
        lbl   = int(labels[b * num_particles])
        color = _label_color(lbl, num_labels)
        cx, cy = xs.mean(), ys.mean()
        for xi, yi in zip(xs, ys):
            ax2.plot([cx, xi], [cy, yi], color=color, alpha=0.12, linewidth=0.6, zorder=1)
        ax2.scatter(xs, ys, c=[color] * len(xs), s=style.marker_size * 1.6,
                   alpha=0.35, linewidths=0, zorder=2)
        ax2.scatter([cx], [cy], c=[color], s=style.marker_size * 8, alpha=0.90,
                   marker="D", linewidths=0.5, edgecolors="white", zorder=3)
    ax2.set_title(f"Posterior spread per sample ({B} samples)", fontsize=style.subtitle_fontsize)
    ax2.set_xlabel("UMAP dim 1", fontsize=style.label_fontsize)
    ax2.set_ylabel("UMAP dim 2", fontsize=style.label_fontsize)
    ax2.tick_params(labelsize=style.tick_fontsize)
    ax2.grid(alpha=style.grid_alpha * 0.6)

    suptitle = (
        f"[{tag.upper()}] UMAP of {num_particles} particles - "
        f"step {data['global_step']}  epoch {data['epoch']}  phase={data['phase']}\n"
        f"sigma_mean={data['sigma_mean']:.4f}  |  "
        f"n_neighbors={data['n_neighbors']}  min_dist={data['min_dist']}  "
        f"metric={data['metric']}  |  {B} samples x {num_particles} particles"
    )
    fig.suptitle(suptitle, fontsize=style.subtitle_fontsize, y=1.02)

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
    s = get_sigma_stats(model)
    pts, lbs = collect_particle_logits(model, probe_batch, cfg)
    data = compute_particle_umap_data(
        pts, lbs, tag=tag, global_step=global_step, epoch=epoch, phase=phase,
        sigma_mean=s["sigma_mean"], cfg=cfg, num_particles=cfg.num_particles,
    )
    save_figure_data(cfg, f"umap_{tag}_step_{global_step:06d}", data)
    plot_particle_umap(data, cfg, style)


def run_and_plot_dual_umap(teacher_model, student_model, teacher_probe, student_probe,
                           global_step, epoch, phase, cfg, style=DEFAULT_STYLE,
                           teacher_tag="teacher", student_tag="student"):
    run_and_plot_umap(teacher_model, teacher_probe, teacher_tag, global_step, epoch, phase, cfg, style)
    run_and_plot_umap(student_model, student_probe, student_tag, global_step, epoch, phase, cfg, style)


# =========================================================================
# GROMOV-WASSERSTEIN  (single-input: 1 cost matrix K x K MOI SAMPLE trong batch)
# =========================================================================

def pairwise_dist_matrix(resp: torch.Tensor, distance: str) -> torch.Tensor:
    """
    resp: [B, K, C] -- K responses (particle) cho moi sample trong batch.
    Tra ve C: [B, K, K] voi C[b,i,j] = d(resp[b,i], resp[b,j]). Day chinh
    la "single-input" cost matrix trong Algorithm 1: moi sample b trong
    batch co MOT cost matrix K x K rieng (khac voi Algorithm 2
    batch-pooled, noi cost matrix la (nK) x (nK)).
    """
    if distance == "cosine":
        resp_n = F.normalize(resp.float(), dim=-1)
        sim = torch.bmm(resp_n, resp_n.transpose(1, 2))     # [B,K,K]
        return 1.0 - sim
    return torch.cdist(resp.float(), resp.float(), p=2.0) ** 2


def compute_entropic_gw(CT, CS, epsilon, sinkhorn_iters, outer_iters):
    """Entropic GW (Eq. 4 trong paper), giai bang Sinkhorn-based proximal-point
    scheme (Peyre et al. 2016), vector hoa qua batch dim B."""
    B, K, _ = CT.shape
    device, dtype = CT.device, torch.float32
    CT = CT.to(dtype)
    CS = CS.to(dtype)
    p     = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)
    q     = p.clone()
    log_p = torch.log(p)
    log_q = torch.log(q)
    CT2p  = torch.bmm(CT ** 2, p.unsqueeze(2)).squeeze(2)

    with torch.no_grad():
        CS_ng   = CS.detach()
        CS2q_ng = torch.bmm(CS_ng ** 2, q.unsqueeze(2)).squeeze(2)
        T = torch.bmm(p.unsqueeze(2), q.unsqueeze(1))
        for _ in range(outer_iters):
            cross       = torch.bmm(CT, torch.bmm(T, CS_ng.transpose(1, 2)))
            cost        = CT2p.unsqueeze(2) + CS2q_ng.unsqueeze(1) - 2.0 * cross
            cost        = cost - cost.amin(dim=(1, 2), keepdim=True)
            log_K_gibbs = torch.clamp(-cost / epsilon, min=-50.0, max=0.0)
            log_a = torch.zeros(B, K, device=device, dtype=dtype)
            log_b = torch.zeros(B, K, device=device, dtype=dtype)
            for _ in range(sinkhorn_iters):
                log_a = log_p - torch.logsumexp(log_K_gibbs + log_b.unsqueeze(1), dim=2)
                log_b = log_q - torch.logsumexp(log_K_gibbs.transpose(1, 2) + log_a.unsqueeze(1), dim=2)
            T = torch.exp(log_a.unsqueeze(2) + log_K_gibbs + log_b.unsqueeze(1))

    CS2q       = torch.bmm(CS ** 2, q.unsqueeze(2)).squeeze(2)
    cross_grad = torch.bmm(CT, torch.bmm(T, CS.transpose(1, 2)))
    cost_grad  = CT2p.unsqueeze(2) + CS2q.unsqueeze(1) - 2.0 * cross_grad
    return (cost_grad * T).sum(dim=(1, 2))         # [B]


def gw_structural_loss(teacher_resp, student_resp, cfg: Config):
    """teacher_resp, student_resp: [B, K, C]. Tra ve scalar = trung binh GW qua B."""
    CT = pairwise_dist_matrix(teacher_resp, cfg.gw_distance).detach()
    CS = pairwise_dist_matrix(student_resp, cfg.gw_distance)
    return compute_entropic_gw(
        CT, CS, cfg.gw_epsilon, cfg.gw_sinkhorn_iters, cfg.gw_outer_iters
    ).mean()


# =========================================================================
# SCHEDULE  (3-phase: task-only -> posterior -> structural)
# =========================================================================

def student_schedule_weights(epoch, cfg: Config):
    n          = cfg.student_num_epochs
    phase1_end = max(1, round(n * cfg.phase1_frac))
    phase2_end = max(phase1_end + 1, round(n * (cfg.phase1_frac + cfg.phase2_frac)))
    phase2_end = min(phase2_end, n - 1) if phase2_end >= n else phase2_end
    if epoch <= phase1_end:
        return 0.0, 0.0
    elif epoch <= phase2_end:
        frac = (epoch - phase1_end) / max(1, phase2_end - phase1_end)
        return cfg.kl_beta_max * min(1.0, frac), 0.0
    else:
        frac = (epoch - phase2_end) / max(1, n - phase2_end)
        return cfg.kl_beta_max, cfg.gw_gamma_max * min(1.0, frac)


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
def evaluate_classification_metrics(model: BLLViTClassifier, loader, device, dtype, num_particles,
                                    ece_num_bins: int = 15):
    """K particles duoc SAMPLE 1 LAN DUY NHAT o batch dau (W co dinh cho ca
    eval set); tra ve posterior-predictive (accuracy/f1), mean-forward
    (accuracy_mean/f1_mean), per-particle max/min, va ECE/ece_mean."""
    model.eval()
    all_labels       = []
    postpred_preds   = []
    mean_preds       = []
    particle_preds   = [[] for _ in range(num_particles)]
    postpred_probs   = []
    mean_probs       = []
    total_loss, n_batches = 0.0, 0
    W = None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
            h = model.backbone_features(images)
            if W is None:
                W = model.head.sample_weights(num_particles)
            particle_logits = torch.einsum("bd,kcd->bkc", h, W)
            mean_logits     = model.head.forward_mean(h)

        particle_logits = particle_logits.float()
        mean_logits     = mean_logits.float()
        B, K, C = particle_logits.shape

        ce = F.cross_entropy(
            particle_logits.reshape(B * K, C),
            labels.unsqueeze(1).expand(-1, K).reshape(-1),
        )
        probs      = F.softmax(particle_logits, dim=-1).mean(dim=1)
        mean_probs_b = F.softmax(mean_logits, dim=-1)
        postpred_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
        mean_preds.extend(torch.argmax(mean_logits, dim=-1).cpu().tolist())
        postpred_probs.append(probs.cpu().numpy())
        mean_probs.append(mean_probs_b.cpu().numpy())
        for k in range(K):
            particle_preds[k].extend(torch.argmax(particle_logits[:, k, :], dim=-1).cpu().tolist())

        all_labels.extend(labels.detach().cpu().tolist())
        total_loss += ce.item()
        n_batches  += 1

    acc_per_particle = [accuracy_score(all_labels, particle_preds[k]) for k in range(num_particles)]
    f1_per_particle  = [f1_score(all_labels, particle_preds[k], average="macro") for k in range(num_particles)]

    all_labels_np     = np.array(all_labels, dtype=np.int64)
    postpred_probs_np = np.concatenate(postpred_probs, axis=0)
    mean_probs_np     = np.concatenate(mean_probs, axis=0)
    ece      = compute_ece(postpred_probs_np, all_labels_np, n_bins=ece_num_bins)
    ece_mean = compute_ece(mean_probs_np, all_labels_np, n_bins=ece_num_bins)

    return {
        "loss":          total_loss / max(1, n_batches),
        "accuracy":      accuracy_score(all_labels, postpred_preds),
        "f1":            f1_score(all_labels, postpred_preds, average="macro"),
        "accuracy_mean": accuracy_score(all_labels, mean_preds),
        "f1_mean":       f1_score(all_labels, mean_preds, average="macro"),
        "accuracy_max":  max(acc_per_particle),
        "accuracy_min":  min(acc_per_particle),
        "f1_max":        max(f1_per_particle),
        "f1_min":        min(f1_per_particle),
        "ece":           ece,
        "ece_mean":      ece_mean,
    }


@torch.no_grad()
def compute_predictive_entropy_scores(model: BLLViTClassifier, loader, cfg: Config,
                                      num_particles: int, max_batches: Optional[int] = None
                                      ) -> np.ndarray:
    """Entropy cua posterior-predictive softmax (trung binh qua K particle)
    cho tung sample -- dung lam OOD score. Entropy cao hon = bat dinh hon."""
    model.eval()
    entropies = []
    for i, (images, _labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(cfg.device, non_blocking=True)
        with torch.autocast(device_type="cuda" if cfg.device == "cuda" else "cpu",
                            dtype=cfg.mixed_precision_dtype):
            logits = model(images, num_particles).float()       # [B,K,C]
        probs = F.softmax(logits, dim=-1).mean(dim=1)            # [B,C]
        ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        entropies.extend(ent.cpu().tolist())
    return np.array(entropies, dtype=np.float64)


def evaluate_ood_auroc(model: BLLViTClassifier, id_loader, ood_loader, cfg: Config,
                       max_batches_each: int = 50) -> dict:
    """AUROC cho ID (CIFAR-100 test) vs OOD (CIFAR-10 test) dua tren
    predictive entropy. label 1 = OOD, score = entropy."""
    id_scores  = compute_predictive_entropy_scores(
        model, id_loader, cfg, cfg.ood_num_particles, max_batches=max_batches_each)
    ood_scores = compute_predictive_entropy_scores(
        model, ood_loader, cfg, cfg.ood_num_particles, max_batches=max_batches_each)
    y_true  = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_score)
    return {
        "auroc": float(auroc),
        "id_scores": id_scores, "ood_scores": ood_scores,
    }


# =========================================================================
# HESSIAN-TRACE SHARPNESS (Hutchinson estimator + double backprop, cung
# tinh than voi PyHessian / Yao et al. 2020). Hessian cua mean-forward CE
# loss tai theta* HIEN TAI (KHONG perturb).
# =========================================================================

def compute_hessian_trace(model: BLLViTClassifier, params: List[torch.Tensor],
                          images: torch.Tensor, labels: torch.Tensor,
                          num_hutchinson_samples: int, seed: Optional[int] = None) -> float:
    """
    Tr(H) ~= (1/M) * sum_i z_i^T H z_i, voi z_i ~ Rademacher i.i.d.
    (Hutchinson, 1990). Grad bac 1 (create_graph=True) chi tinh 1 LAN cho
    ca M mau; moi z_i chi can 1 lan backward bac 2 them. Generator rieng
    (khong dung torch.manual_seed toan cuc) de khong lam xao tron RNG
    stream chinh cua training loop.
    """
    logits = model.forward_mean(images)
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


def evaluate_hessian_trace_sharpness(model: BLLViTClassifier, loader, cfg: Config,
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
# STAGE 1: TEACHER FIT  (pretrained ViT-Large backbone + BLL head)
# =========================================================================

def fit_teacher(cfg: Config) -> Tuple[str, List[Dict]]:
    """
    Teacher = ViT-Large PRETRAINED (timm, ImageNet weights) + BLL head moi
    khoi tao. Backbone pretrained co the duoc finetune nhe (lr nho hon,
    cfg.teacher_backbone_lr_mult) hoac freeze hoan toan
    (teacher_finetune_backbone=False), trong khi BLL head luon hoc tu dau
    qua ELBO. So epoch fit teacher = so epoch distill student (CFG.teacher_num_epochs
    da duoc gan = CFG.student_num_epochs ngay sau khi tao CFG).

    LUU Y (fix bug UMAP bi de): tag UMAP cua giai doan nay la "teacher_fit"
    (KHONG phai "teacher" trung voi giai doan distill) -- global_step o day
    la mot counter rieng, doc lap voi global_step trong distill_student().
    Neu dung chung tag "teacher", file
    figures/umap/teacher_step_000100.png va figure_data/umap_teacher_step_000100.*
    se bi GHI DE boi UMAP cua teacher (frozen) trong giai doan distill khi
    hai counter cung cham step 100, 200, ... -- lam mat vinh vien UMAP cua
    teacher luc dang fit.
    """
    print(f"\n{'='*80}\nSTAGE 1: FITTING TEACHER BLL (pretrained backbone) -- {cfg.teacher_name}\n{'='*80}")

    model = BLLViTClassifier.from_timm_name(
        cfg.teacher_name, cfg.num_labels, cfg.teacher_pretrained,
        cfg.prior_std, cfg.init_log_sigma,
    )
    model.to(cfg.device)
    if cfg.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if not cfg.teacher_finetune_backbone:
        for p in model.backbone.parameters():
            p.requires_grad_(False)

    print(f"[UMAP] Building teacher probe batch ({cfg.umap_probe_samples} samples)...")
    umap_probe = build_umap_probe_batch(cfg, dataset_name="cifar100")

    train_loader, eval_loader = build_cifar100_loaders(cfg, cfg.batch_size)
    n_train = len(train_loader.dataset)

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
    kl_warmup_steps = max(1, int(total_steps * cfg.teacher_kl_warmup_frac))

    s0 = get_sigma_stats(model)
    print(f"[sigma init] mean={s0['sigma_mean']:.5f}  min={s0['sigma_min']:.5f}  max={s0['sigma_max']:.5f}")

    history: List[Dict] = []
    model.train()
    global_step      = 0
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "teacher_vit_large_bll_latest")

    epoch_pbar = tqdm(range(1, cfg.teacher_num_epochs + 1), desc="teacher epochs", unit="epoch")
    for epoch in epoch_pbar:
        epoch_ce, epoch_kl, epoch_kl_term, epoch_total, n_steps = 0.0, 0.0, 0.0, 0.0, 0
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
                logits = model(images, cfg.num_particles)
            logits = logits.float()
            B, K, C = logits.shape
            ce      = F.cross_entropy(logits.reshape(B*K, C),
                                      labels.unsqueeze(1).expand(-1, K).reshape(-1))
            kl_raw  = model.kl_divergence()
            kl_beta = cfg.teacher_kl_beta_max * min(1.0, global_step / kl_warmup_steps)
            kl_scale = cfg.batch_size / max(1, n_train)
            kl_term   = kl_scale * kl_beta * kl_raw
            total_raw = ce + kl_term
            loss = total_raw / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.umap_every_n_steps == 0:
                    # FIX: tag "teacher_fit" (truoc la "teacher") de KHONG
                    # trung voi tag cua teacher ben giai doan distill_student().
                    run_and_plot_umap(model, umap_probe, "teacher_fit", global_step, epoch, "fit", cfg)
                    model.train()

            epoch_ce      += ce.item()
            epoch_kl      += kl_raw.item()
            epoch_kl_term += kl_term.item()
            epoch_total   += total_raw.item()
            n_steps       += 1

            s = get_sigma_stats(model)
            step_pbar.set_postfix(ce=f"{ce.item():.4f}", kl=f"{kl_raw.item():.1f}",
                                  kl_beta=f"{kl_beta:.3f}", sig_m=f"{s['sigma_mean']:.4f}")

        avg_ce      = epoch_ce / max(1, n_steps)
        avg_kl      = epoch_kl / max(1, n_steps)
        avg_kl_term = epoch_kl_term / max(1, n_steps)
        avg_total   = epoch_total / max(1, n_steps)
        metrics = evaluate_classification_metrics(
            model, eval_loader, cfg.device, cfg.mixed_precision_dtype, cfg.eval_num_particles,
            ece_num_bins=cfg.ece_num_bins)
        elapsed = time.time() - t0
        s = get_sigma_stats(model)

        epoch_pbar.set_postfix(ce=f"{avg_ce:.4f}", test_acc=f"{metrics['accuracy']:.4f}",
                               sig_m=f"{s['sigma_mean']:.4f}")

        record = {
            "model": "teacher", "epoch": epoch, "ce_loss": avg_ce, "kl_raw": avg_kl,
            "kl_term": avg_kl_term, "total_loss": avg_total, "eval_loss": metrics["loss"],
            "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "accuracy_mean": metrics["accuracy_mean"], "accuracy_max": metrics["accuracy_max"],
            "accuracy_min": metrics["accuracy_min"], "f1_mean": metrics["f1_mean"],
            "f1_max": metrics["f1_max"], "f1_min": metrics["f1_min"],
            "sigma_mean": s["sigma_mean"], "sigma_min": s["sigma_min"], "sigma_max": s["sigma_max"],
            "epoch_time_sec": elapsed,
        }
        log_jsonl(cfg.log_file, record)
        history.append({
            "epoch": epoch, "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "accuracy_mean": metrics["accuracy_mean"], "accuracy_max": metrics["accuracy_max"],
            "accuracy_min": metrics["accuracy_min"], "f1_mean": metrics["f1_mean"],
            "f1_max": metrics["f1_max"], "f1_min": metrics["f1_min"],
            "total_loss": avg_total, "ce_loss": avg_ce, "kl_loss": avg_kl_term,
        })

        save_bll_checkpoint(model, cfg, latest_ckpt_path)
        print(f"[latest teacher] epoch={epoch}  eval_loss={metrics['loss']:.4f}"
              f"  acc_mean={metrics['accuracy_mean']:.4f}"
              f"  acc[min,max]=[{metrics['accuracy_min']:.4f},{metrics['accuracy_max']:.4f}]"
              f"  sigma_mean={s['sigma_mean']:.4f}")
        model.train()

    ckpt_path = os.path.join(cfg.checkpoint_dir, "teacher_vit_large_bll")
    save_bll_checkpoint(model, cfg, ckpt_path)

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path, history


# =========================================================================
# STAGE 2: STUDENT DISTILLATION  (ViT-Small, random init, single-input GW)
# =========================================================================

def distill_student(cfg: Config, teacher_ckpt_path: str, teacher_model_for_umap: BLLViTClassifier
                    ) -> Tuple[str, List[Dict]]:
    """Student = ViT-Small KHONG pretrained (distill tu dau). 3-phase
    schedule + ELBO + GW single-input, giong logic ban CNN.

    LUU Y (fix bug UMAP bi de): tag UMAP cua teacher/student trong giai
    doan nay la "teacher_distill"/"student_distill" (truoc la "teacher"/
    "student"). global_step o day la mot counter rieng, doc lap voi
    global_step trong fit_teacher() -- ca hai deu bat dau tu 0 va cung
    cham cfg.umap_every_n_steps, nen neu dung chung tag "teacher" thi UMAP
    cua teacher luc fit (stage 1) se bi GHI DE boi UMAP cua teacher
    (frozen) o stage nay khi global_step trung nhau (100, 200, ...).
    """
    print(f"\n{'='*80}\nSTAGE 2: DISTILLING STUDENT -- {cfg.student_name}\n{'='*80}")

    student = BLLViTClassifier.from_timm_name(
        cfg.student_name, cfg.num_labels, cfg.student_pretrained,
        cfg.prior_std, cfg.init_log_sigma,
    )
    student.to(cfg.device)
    if cfg.use_gradient_checkpointing:
        student.enable_gradient_checkpointing()

    # Teacher dung de tinh logits distill (frozen, tu checkpoint).
    teacher = load_bll_checkpoint(teacher_ckpt_path, cfg.teacher_name)
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
    n_train = len(train_loader.dataset)

    optimizer    = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps  = (len(train_loader) // cfg.gradient_accumulation_steps) * cfg.student_num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(1, warmup_steps)) *
                     max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )

    s0 = get_sigma_stats(student)
    print(f"[sigma init] mean={s0['sigma_mean']:.5f}  min={s0['sigma_min']:.5f}  max={s0['sigma_max']:.5f}")
    print(f"[student] K={cfg.num_particles}  batch={cfg.distill_batch_size}  epochs={cfg.student_num_epochs}")

    history: List[Dict] = []
    student.train()
    global_step      = 0
    latest_ckpt_path = os.path.join(cfg.checkpoint_dir, "student_vit_small_bll_gw_latest")

    epoch_pbar = tqdm(range(1, cfg.student_num_epochs + 1), desc="student (BLL+GW) epochs", unit="epoch")
    for epoch in epoch_pbar:
        beta_kl, gamma = student_schedule_weights(epoch, cfg)
        phase = ("task-only" if gamma == 0 and beta_kl == 0 else
                 "posterior" if gamma == 0 else "structural")
        print(f"\n[schedule] epoch {epoch}/{cfg.student_num_epochs}  "
              f"phase={phase}  beta_kl={beta_kl:.3f}  gamma={gamma:.3f}")

        epoch_ce, epoch_kl, epoch_gw, n_steps = 0.0, 0.0, 0.0, 0
        epoch_kl_term, epoch_gw_term, epoch_total = 0.0, 0.0, 0.0
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
                teacher_logits = teacher(images, cfg.num_particles).float()  # [B,K,C]

            with torch.autocast(device_type="cuda" if cfg.device == "cuda" else "cpu",
                                dtype=cfg.mixed_precision_dtype):
                student_logits = student(images, cfg.num_particles)
            student_logits = student_logits.float()
            B, K, C = student_logits.shape

            ce       = F.cross_entropy(student_logits.reshape(B*K, C),
                                       labels.unsqueeze(1).expand(-1, K).reshape(-1))
            kl_raw   = student.kl_divergence()
            kl_scale = cfg.distill_batch_size / max(1, n_train)
            # GW SINGLE-INPUT: cost matrix K x K rieng cho TUNG sample (Algorithm 1).
            gw_loss  = (gw_structural_loss(teacher_logits, student_logits, cfg)
                       if gamma > 0.0 else torch.zeros((), device=cfg.device))

            kl_term   = kl_scale * beta_kl * kl_raw
            gw_term   = gamma * gw_loss
            total_raw = ce + kl_term + gw_term
            loss = total_raw / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.umap_every_n_steps == 0:
                    # FIX: tag rieng cho stage distill ("teacher_distill" /
                    # "student_distill") de KHONG trung voi tag "teacher_fit"
                    # cua stage 1, du global_step cua 2 stage co the trung nhau.
                    run_and_plot_dual_umap(
                        teacher_model=teacher_model_for_umap, student_model=student,
                        teacher_probe=teacher_umap_probe, student_probe=student_umap_probe,
                        global_step=global_step, epoch=epoch, phase=phase, cfg=cfg,
                        teacher_tag="teacher_distill", student_tag="student_distill",
                    )
                    student.train()

            epoch_ce      += ce.item()
            epoch_kl      += kl_raw.item()
            epoch_gw      += float(gw_loss.item())
            epoch_kl_term += float(kl_term.item())
            epoch_gw_term += float(gw_term.item())
            epoch_total   += float(total_raw.item())
            n_steps       += 1

            s = get_sigma_stats(student)
            step_pbar.set_postfix(ce=f"{ce.item():.4f}", kl=f"{kl_raw.item():.1f}",
                                  gw=f"{float(gw_loss.item()):.4f}", sig_m=f"{s['sigma_mean']:.4f}")

        avg_ce      = epoch_ce / max(1, n_steps)
        avg_kl      = epoch_kl / max(1, n_steps)
        avg_gw      = epoch_gw / max(1, n_steps)
        avg_kl_term = epoch_kl_term / max(1, n_steps)
        avg_gw_term = epoch_gw_term / max(1, n_steps)
        avg_total   = epoch_total / max(1, n_steps)
        metrics = evaluate_classification_metrics(
            student, eval_loader, cfg.device, cfg.mixed_precision_dtype, cfg.eval_num_particles,
            ece_num_bins=cfg.ece_num_bins)
        elapsed = time.time() - t0
        s = get_sigma_stats(student)

        epoch_pbar.set_postfix(ce=f"{avg_ce:.4f}", gw=f"{avg_gw:.4f}",
                               test_acc=f"{metrics['accuracy']:.4f}", sig_m=f"{s['sigma_mean']:.4f}")

        record = {
            "model": "student", "epoch": epoch, "phase": phase, "beta_kl": beta_kl, "gamma": gamma,
            "ce_loss": avg_ce, "kl_raw": avg_kl, "gw_loss": avg_gw, "kl_term": avg_kl_term,
            "gw_term": avg_gw_term, "total_loss": avg_total, "eval_loss": metrics["loss"],
            "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "accuracy_mean": metrics["accuracy_mean"], "accuracy_max": metrics["accuracy_max"],
            "accuracy_min": metrics["accuracy_min"], "f1_mean": metrics["f1_mean"],
            "f1_max": metrics["f1_max"], "f1_min": metrics["f1_min"],
            "sigma_mean": s["sigma_mean"], "sigma_min": s["sigma_min"], "sigma_max": s["sigma_max"],
            "epoch_time_sec": elapsed,
        }
        log_jsonl(cfg.log_file, record)
        history.append({
            "epoch": epoch, "accuracy": metrics["accuracy"], "f1": metrics["f1"],
            "accuracy_mean": metrics["accuracy_mean"], "accuracy_max": metrics["accuracy_max"],
            "accuracy_min": metrics["accuracy_min"], "f1_mean": metrics["f1_mean"],
            "f1_max": metrics["f1_max"], "f1_min": metrics["f1_min"],
            "total_loss": avg_total, "ce_loss": avg_ce,
            "kl_loss": avg_kl_term, "gw_loss": avg_gw_term,
        })

        save_bll_checkpoint(student, cfg, latest_ckpt_path)
        print(f"[latest student] epoch={epoch}  eval_loss={metrics['loss']:.4f}"
              f"  acc_mean={metrics['accuracy_mean']:.4f}"
              f"  acc[min,max]=[{metrics['accuracy_min']:.4f},{metrics['accuracy_max']:.4f}]"
              f"  sigma_mean={s['sigma_mean']:.4f}")
        student.train()

    ckpt_path = os.path.join(cfg.checkpoint_dir, "student_vit_small_bll_gw")
    save_bll_checkpoint(student, cfg, ckpt_path)

    del student, teacher, teacher_model_for_umap, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ckpt_path, history


# =========================================================================
# LOSS LANDSCAPE
# =========================================================================
# ViT dung LayerNorm (khong co running stats), nen KHONG can context-manager
# dac biet nhu _bn_use_batch_stats_context o ban CNN. LayerNorm affine
# (1 chieu) bi zero-out tu dong boi dieu kien "dim() <= 1" trong
# filter_normalize_direction(), giong het bias.

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
def evaluate_classification_loss(model: BLLViTClassifier, loader, device, dtype, max_batches=None):
    """Mean-forward CE loss tren mot so batch co dinh, dung de quet loss
    landscape. Khong can xu ly gi dac biet cho LayerNorm (khac BatchNorm)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
            logits = model.forward_mean(images)
            loss   = F.cross_entropy(logits.float(), labels)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(20.0)
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(1, n_batches)


def compute_loss_landscape(model: BLLViTClassifier, loader, cfg: Config, seed=0, raw_dirs=None):
    device = cfg.device
    grid   = cfg.landscape_grid_size
    torch.manual_seed(seed)
    named       = list(model.landscape_named_parameters())
    params      = [p for _, p in named]
    base_params = [p.detach().clone() for p in params]

    # Sử dụng explicit directions từ shared_dirs nếu có
    if raw_dirs is not None:
        raw_dir1, raw_dir2 = raw_dirs
        # CLONE trước khi normalize để không làm hỏng raw_dirs dùng chung cho phase sau
        dir1 = [d.clone().to(p.device) for d, p in zip(raw_dir1, params)]
        dir2 = [d.clone().to(p.device) for d, p in zip(raw_dir2, params)]
    else:
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

def compute_landscape_data(teacher_model, student_model, cfg: Config, tag: str, shared_dirs: dict = None) -> dict:
    landscape_loader = build_landscape_loader(cfg)
    results = {}
    for key, model in [(TEACHER_KEY, teacher_model), (STUDENT_KEY, student_model)]:
        model.to(cfg.device)
        raw_dirs = shared_dirs[key] if shared_dirs and key in shared_dirs else None
        print(f"\n[landscape:{tag}] Sweeping {cfg.landscape_grid_size}x{cfg.landscape_grid_size} for {key} ...")
        alphas, betas, loss_grid = compute_loss_landscape(
            model, landscape_loader, cfg, seed=cfg.landscape_seed, raw_dirs=raw_dirs)
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


def _draw_3d_axis(ax, A, B, lg, key, fig, style: PlotStyle, vmin: float, vmax: float):
    """Mau va truc Z deu theo thang LINEAR (gia tri loss KHONG clip)."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    facecolors = plt.get_cmap(style.cmap_name)(norm(lg))
    surf = ax.plot_surface(A, B, lg, facecolors=facecolors, linewidth=0,
                           antialiased=True, edgecolor="none", shade=False)
    ax.set_title(f"{key} -- 3D (linear scale)", fontsize=style.subtitle_fontsize)
    ax.set_xlabel("alpha", fontsize=style.label_fontsize)
    ax.set_ylabel("beta", fontsize=style.label_fontsize)
    ax.set_zlabel("loss", fontsize=style.label_fontsize)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=style.cmap_name)
    mappable.set_array(lg)
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="loss")


def _draw_2d_axis(ax, A, B, lg, key, fig, style: PlotStyle, vmin: float, vmax: float):
    """Contour 2D voi thang mau LINEAR, su dung vmin, vmax chung."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 20)
    cs = ax.contourf(A, B, lg, levels=levels, cmap=style.cmap_name, norm=norm)
    ax.contour(A, B, lg, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
    ax.set_title(f"{key} -- 2D (linear scale)", fontsize=style.subtitle_fontsize)
    ax.set_xlabel("alpha", fontsize=style.label_fontsize)
    ax.set_ylabel("beta", fontsize=style.label_fontsize)
    ax.scatter([0], [0], color="red", marker="*", s=150, label="theta* (posterior mean)")
    ax.legend(loc="upper right", fontsize=style.legend_fontsize)
    fig.colorbar(cs, ax=ax, shrink=0.9, label="loss")


def plot_landscapes_all(data: dict, cfg: Config, title: str, save_name_prefix: str,
                        style: PlotStyle = DEFAULT_STYLE):
    """Ve tu du lieu landscape da tinh san, dung CHUNG thang do heatmap (vmin, vmax)."""
    results = data["results"]
    keys = list(results.keys())
    n    = len(keys)
    
    # Tính vmin, vmax chung cho toàn bộ grids để đồng bộ thang đo màu
    global_vmin = min([float(r["loss_grid"].min()) for r in results.values()])
    global_vmax = max([float(r["loss_grid"].max()) for r in results.values()])
    if global_vmax <= global_vmin:
        global_vmax = global_vmin + 1.0

    for suffix, rows_spec in [("_3d_1x2.png", [["3d"]]), ("_2d_1x2.png", [["2d"]]),
                              ("_2x2.png", [["3d"], ["2d"]])]:
        fig = plt.figure(figsize=(style.figsize_single[0]*n, style.figsize_single[1]*len(rows_spec)))
        for r, row in enumerate(rows_spec):
            mode = row[0]
            for c, key in enumerate(keys):
                rdata = results[key]
                alphas, betas, lg = rdata["alphas"], rdata["betas"], rdata["loss_grid"]
                A, B = np.meshgrid(alphas, betas, indexing="ij")
                idx  = r * n + c + 1
                if mode == "3d":
                    ax = fig.add_subplot(len(rows_spec), n, idx, projection="3d")
                    _draw_3d_axis(ax, A, B, lg, key, fig, style, global_vmin, global_vmax)
                else:
                    ax = fig.add_subplot(len(rows_spec), n, idx)
                    _draw_2d_axis(ax, A, B, lg, key, fig, style, global_vmin, global_vmax)
        fig.suptitle(title, fontsize=style.title_fontsize, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = os.path.join(cfg.figure_dir, save_name_prefix + suffix)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=style.dpi)
        plt.close(fig)
        print(f"[saved figure] {p}")


def run_landscape_and_plot(teacher_model, student_model, cfg: Config, tag: str,
                           title: str, save_name_prefix: str, shared_dirs: dict = None,
                           style: PlotStyle = DEFAULT_STYLE):
    data = compute_landscape_data(teacher_model, student_model, cfg, tag, shared_dirs)
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
    """Accuracy/F1 vs epoch: duong "mean" (mu-forward, theta*) + vung mo
    [min, max] (do trai hieu suat giua cac particle)."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_wide)

    for key, label, color in [
        (history.get("teacher_key", TEACHER_KEY), "ViT-Large teacher (BLL)", style.teacher_color),
        (history.get("student_key", STUDENT_KEY), "ViT-Small student (BLL+GW)", style.student_color),
    ]:
        rows = history.get("teacher_history" if "teacher" in key.lower() else "student_history", [])
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]

        acc_mean = [r["accuracy_mean"] for r in rows]
        acc_max  = [r["accuracy_max"]  for r in rows]
        acc_min  = [r["accuracy_min"]  for r in rows]
        axes[0].plot(epochs, acc_mean, marker="o", linewidth=style.line_width,
                    markersize=style.marker_size, label=f"{label} - mean (mu)", color=color, zorder=3)
        axes[0].fill_between(epochs, acc_min, acc_max, color=color, alpha=style.band_alpha,
                             label=f"{label} - particle [min, max]", zorder=1)

        f1_mean = [r["f1_mean"] for r in rows]
        f1_max  = [r["f1_max"]  for r in rows]
        f1_min  = [r["f1_min"]  for r in rows]
        axes[1].plot(epochs, f1_mean, marker="o", linewidth=style.line_width,
                    markersize=style.marker_size, label=f"{label} - mean (mu)", color=color, zorder=3)
        axes[1].fill_between(epochs, f1_min, f1_max, color=color, alpha=style.band_alpha,
                             label=f"{label} - particle [min, max]", zorder=1)

    for ax, title, ylabel in [
        (axes[0], "Test Accuracy vs Epoch (CIFAR-100)\n(line = mu-forward, band = particle min-max)", "Accuracy"),
        (axes[1], "Test F1 (macro) vs Epoch (CIFAR-100)\n(line = mu-forward, band = particle min-max)", "F1 Score"),
    ]:
        ax.set_title(title, fontsize=style.subtitle_fontsize)
        ax.set_xlabel("Epoch", fontsize=style.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=style.label_fontsize)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=style.tick_fontsize)
        ax.grid(alpha=style.grid_alpha)
        ax.legend(fontsize=style.legend_fontsize, loc="lower right")
    fig.suptitle("CIFAR-100 Classification: BLL Teacher (ViT-Large) vs BLL+GW Student (ViT-Small)",
                fontsize=style.title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(cfg.figure_dir, "accuracy_f1_vs_epoch.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=style.dpi)
    plt.close(fig)
    print(f"[saved figure] {p}")


def plot_loss_curves(history: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    panels = [
        ("total_loss", "Total Loss vs Epoch", "Total loss"),
        ("ce_loss",    "Cross-Entropy Loss vs Epoch", "CE loss"),
        ("kl_loss",    "KL Loss (weighted) vs Epoch", "kl_scale . beta_KL . KL"),
        ("gw_loss",    "GW Structural Loss (weighted) vs Epoch", "gamma . GW"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=style.figsize_grid2x2)
    flat_axes = axes.reshape(-1)

    teacher_rows = history.get("teacher_history", [])
    student_rows = history.get("student_history", [])

    for ax, (field, title, ylabel) in zip(flat_axes, panels):
        plotted_any = False
        for label, color, rows in [
            ("ViT-Large teacher (BLL)", style.teacher_color, teacher_rows),
            ("ViT-Small student (BLL+GW)", style.student_color, student_rows),
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

    fig.suptitle("Loss Components vs Epoch: BLL Teacher (ViT-Large) vs BLL+GW Student (ViT-Small)",
                fontsize=style.title_fontsize, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(cfg.figure_dir, "loss_curves.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=style.dpi)
    plt.close(fig)
    print(f"[saved figure] {p}")


# =========================================================================
# OOD PLOT  (CIFAR-100 ID vs CIFAR-10 OOD, predictive entropy + AUROC)
# =========================================================================

def run_and_save_ood_eval(teacher_model, student_model, cfg: Config) -> dict:
    """Tinh AUROC OOD (CIFAR-10), ECE (CIFAR-100 test, Naeini/Guo) va
    Hessian-trace sharpness (Hutchinson estimator, tren build_landscape_loader)
    cho ca teacher va student, luu du lieu tho.

    Truoc day ham nay CHI tinh AUROC -- compute_ece() va
    evaluate_hessian_trace_sharpness() da ton tai trong file nhung khong
    duoc goi o Step 4B. Gio ca 3 metric deu duoc tinh cho model FINAL
    (checkpoint da load), khop voi Proposition 5.3 (necessary-condition gap
    can accuracy/ECE/sharpness, khong chi AUROC)."""
    id_loader        = build_cifar100_loaders(cfg, cfg.batch_size)[1]   # eval split
    ood_loader       = build_cifar10_ood_loader(cfg, cfg.batch_size)
    landscape_loader = build_landscape_loader(cfg)                      # cho Hessian-trace

    results = {}
    for key, model in [(TEACHER_KEY, teacher_model), (STUDENT_KEY, student_model)]:
        model.to(cfg.device)

        # ---- AUROC (OOD detection: ID=CIFAR-100 vs OOD=CIFAR-10) ----
        print(f"\n[OOD] Evaluating {key} on ID=CIFAR-100 test vs OOD=CIFAR-10 test ...")
        r = evaluate_ood_auroc(model, id_loader, ood_loader, cfg)
        print(f"[OOD] {key}: AUROC = {r['auroc']:.4f}")

        # ---- ECE (calibration, ID=CIFAR-100 test) ----
        print(f"[ECE] Evaluating {key} calibration on ID=CIFAR-100 test ...")
        cls_metrics = evaluate_classification_metrics(
            model, id_loader, cfg.device, cfg.mixed_precision_dtype,
            cfg.eval_num_particles, ece_num_bins=cfg.ece_num_bins)
        r["ece"]      = cls_metrics["ece"]        # posterior-predictive (K particles)
        r["ece_mean"] = cls_metrics["ece_mean"]   # mean-forward (W = mu)
        print(f"[ECE] {key}: ECE = {r['ece']:.4f}  (ECE mean-forward = {r['ece_mean']:.4f})")

        # ---- Hessian-trace sharpness (Hutchinson, double-backprop) ----
        print(f"[Hessian] Estimating {key} Hessian-trace sharpness "
              f"({cfg.hessian_num_hutchinson_samples} Hutchinson samples x "
              f"{cfg.hessian_eval_batches} batches) ...")
        r["hessian_trace"] = evaluate_hessian_trace_sharpness(
            model, landscape_loader, cfg, seed=cfg.hessian_seed)
        print(f"[Hessian] {key}: Tr(H) ~= {r['hessian_trace']:.4f}")

        results[key] = r

    payload = {}
    for key, r in results.items():
        safe_key = key.split(" ")[0].replace("-", "_").lower()
        payload[f"{safe_key}_id_scores"]     = np.array(r["id_scores"])
        payload[f"{safe_key}_ood_scores"]    = np.array(r["ood_scores"])
        payload[f"{safe_key}_auroc"]         = r["auroc"]
        payload[f"{safe_key}_ece"]           = r["ece"]
        payload[f"{safe_key}_ece_mean"]      = r["ece_mean"]
        payload[f"{safe_key}_hessian_trace"] = r["hessian_trace"]
        payload[f"{safe_key}_model_key"]     = key
    save_figure_data(cfg, "ood_eval", payload)

    summary = {
        key: {
            "auroc":         r["auroc"],
            "ece":           r["ece"],
            "ece_mean":      r["ece_mean"],
            "hessian_trace": r["hessian_trace"],
        }
        for key, r in results.items()
    }
    return summary, payload


def plot_ood_eval(payload: dict, cfg: Config, style: PlotStyle = DEFAULT_STYLE):
    """Histogram predictive-entropy ID vs OOD cho teacher va student (2
    subplot), kem AUROC trong title."""
    fig, axes = plt.subplots(1, 2, figsize=style.figsize_wide)
    for ax, prefix, label, color in [
        (axes[0], "vit_large", "ViT-Large teacher (BLL)", style.teacher_color),
        (axes[1], "vit_small", "ViT-Small student (BLL+GW)", style.student_color),
    ]:
        id_scores     = payload.get(f"{prefix}_id_scores")
        ood_scores    = payload.get(f"{prefix}_ood_scores")
        auroc         = payload.get(f"{prefix}_auroc")
        ece           = payload.get(f"{prefix}_ece")
        hessian_trace = payload.get(f"{prefix}_hessian_trace")
        if id_scores is None or ood_scores is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.hist(id_scores, bins=30, alpha=0.6, density=True, color="tab:green",
               label="ID: CIFAR-100 test")
        ax.hist(ood_scores, bins=30, alpha=0.6, density=True, color="tab:red",
               label="OOD: CIFAR-10 test")
        auroc_str = f"{float(auroc):.4f}" if auroc is not None else "n/a"
        ece_str   = f"{float(ece):.4f}" if ece is not None else "n/a"
        htr_str   = f"{float(hessian_trace):.3g}" if hessian_trace is not None else "n/a"
        ax.set_title(
            f"{label}\nAUROC = {auroc_str}   |   ECE = {ece_str}   |   Tr(H) \u2248 {htr_str}",
            fontsize=style.subtitle_fontsize)
        ax.set_xlabel("Predictive entropy (posterior-predictive)", fontsize=style.label_fontsize)
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

    # ============================================================
    # TẠO GLOBAL SHARED DIRECTIONS ĐỂ ĐẢM BẢO BEFORE/AFTER CÙNG DIRECTION
    # ============================================================
    shared_dirs = {}
    if CFG.run_pre_landscape or CFG.run_post_landscape:
        print("\n>>> Generating fixed random directions for landscape (Before & After) <<<")
        torch.manual_seed(CFG.landscape_seed)
        t_dummy = BLLViTClassifier.from_timm_name(CFG.teacher_name, CFG.num_labels, False, CFG.prior_std, CFG.init_log_sigma)
        s_dummy = BLLViTClassifier.from_timm_name(CFG.student_name, CFG.num_labels, False, CFG.prior_std, CFG.init_log_sigma)
        
        shared_dirs[TEACHER_KEY] = (
            get_random_direction_like([p for _, p in t_dummy.landscape_named_parameters()]),
            get_random_direction_like([p for _, p in t_dummy.landscape_named_parameters()])
        )
        shared_dirs[STUDENT_KEY] = (
            get_random_direction_like([p for _, p in s_dummy.landscape_named_parameters()]),
            get_random_direction_like([p for _, p in s_dummy.landscape_named_parameters()])
        )
        del t_dummy, s_dummy
    # ============================================================

    teacher_ckpt_path = os.path.join(CFG.checkpoint_dir, "teacher_vit_large_bll")
    teacher_history: List[Dict] = []

    if bll_checkpoint_exists(teacher_ckpt_path):
        print(">>> Teacher checkpoint found. Skipping teacher fit.")
    else:
        if CFG.run_pre_landscape:
            print("\n>>> STEP 1: Loss landscape BEFORE distillation (random/pretrained init) <<<")
            t_fresh = BLLViTClassifier.from_timm_name(
                CFG.teacher_name, CFG.num_labels, CFG.teacher_pretrained,
                CFG.prior_std, CFG.init_log_sigma)
            s_fresh = BLLViTClassifier.from_timm_name(
                CFG.student_name, CFG.num_labels, CFG.student_pretrained,
                CFG.prior_std, CFG.init_log_sigma)
            run_landscape_and_plot(
                t_fresh, s_fresh, CFG, tag="BEFORE",
                title="Loss Landscape BEFORE Knowledge Distillation",
                save_name_prefix="loss_landscape_BEFORE_distillation",
                shared_dirs=shared_dirs, # <--- TRUYỀN DIR VÀO ĐÂY
            )
            del t_fresh, s_fresh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if CFG.run_teacher_fit:
            print(f"\n>>> STEP 2: Fitting teacher BLL on pretrained ViT-Large "
                  f"({CFG.teacher_num_epochs} epochs, = student_num_epochs) <<<")
            teacher_ckpt_path, teacher_history = fit_teacher(CFG)

    student_ckpt_path = os.path.join(CFG.checkpoint_dir, "student_vit_small_bll_gw")
    student_history: List[Dict] = []

    if CFG.run_distillation:
        print(f"\n>>> STEP 3: Distilling student (BLL+GW single-input, "
              f"{CFG.student_num_epochs} epochs) <<<")
        print(f"[distill] Loading teacher checkpoint for UMAP: {teacher_ckpt_path}")
        teacher_for_umap = load_bll_checkpoint(teacher_ckpt_path, CFG.teacher_name)
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
        teacher_model = load_bll_checkpoint(teacher_ckpt_path, CFG.teacher_name)
        student_model = load_bll_checkpoint(student_ckpt_path, CFG.student_name)
        run_landscape_and_plot(
            teacher_model, student_model, CFG, tag="AFTER",
            title="Loss Landscape AFTER Knowledge Distillation",
            save_name_prefix="loss_landscape_AFTER_distillation",
            shared_dirs=shared_dirs, # <--- TRUYỀN DIR VÀO ĐÂY
        )

        if CFG.run_ood_eval:
            print("\n>>> STEP 4b: OOD AUROC (ID=CIFAR-100, OOD=CIFAR-10) "
                  "+ ECE + Hessian-trace sharpness (final checkpoints) <<<")
            ood_summary, ood_payload = run_and_save_ood_eval(teacher_model, student_model, CFG)
            for key, m in ood_summary.items():
                print(f"[Step 4b summary] {key}: "
                      f"AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  "
                      f"ECE_mean={m['ece_mean']:.4f}  Tr(H)~={m['hessian_trace']:.4f}")
            plot_ood_eval(ood_payload, CFG)

        del teacher_model, student_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if CFG.run_metric_curves and (teacher_history or student_history):
        print("\n>>> STEP 5: Plotting accuracy / F1 curves <<<")
        hist_payload = {"teacher_history": teacher_history, "student_history": student_history,
                        "teacher_key": TEACHER_KEY, "student_key": STUDENT_KEY}
        plot_metric_curves(hist_payload, CFG)

        print("\n>>> STEP 6: Plotting loss curves (total / CE / KL / GW) <<<")
        plot_loss_curves(hist_payload, CFG)

    print("\nFull pipeline complete.")
    print(f"Figures      : {CFG.figure_dir}")
    print(f"Figure data  : {CFG.figure_data_dir}  (dung de ve lai, xem replot_all_from_saved())")
    print(f"Checkpoints  : {CFG.checkpoint_dir}")
    print(f"Log          : {CFG.log_file}")


if __name__ == "__main__":
    main()