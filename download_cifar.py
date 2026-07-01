"""
download_cifar.py

Tai CIFAR-10 va CIFAR-100 tu nguon goc (cs.toronto.edu) va luu du lieu
+ nhan vao dataset/cifar-10 va dataset/cifar-100 duoi dang file .npy
(de load nhanh, khong can giai nen lai moi lan train).

Cau truc output:
    dataset/cifar-10/
        train_images.npy   (50000, 32, 32, 3) uint8
        train_labels.npy   (50000,)           int64
        test_images.npy    (10000, 32, 32, 3) uint8
        test_labels.npy    (10000,)           int64
        classes.txt        ten 10 class

    dataset/cifar-100/
        train_images.npy        (50000, 32, 32, 3) uint8
        train_labels.npy        (50000,)  fine label, int64
        train_coarse_labels.npy (50000,)  coarse label, int64
        test_images.npy / test_labels.npy / test_coarse_labels.npy
        fine_classes.txt     ten 100 class
        coarse_classes.txt   ten 20 superclass

Su dung:
    python download_cifar.py                  # tai ca 2 bo (mac dinh 8 luong song song)
    python download_cifar.py --which cifar10   # chi tai CIFAR-10
    python download_cifar.py --which cifar100  # chi tai CIFAR-100
    python download_cifar.py --connections 16  # tang so luong de tai nhanh hon
    python download_cifar.py --connections 1   # tai tuan tu (1 luong) neu mang khong on dinh
    python download_cifar.py --keep-raw        # giu lai file .tar.gz / file giai nen
"""

import argparse
import hashlib
import pickle
import shutil
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Cau hinh duong dan & nguon tai
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "_raw"

CIFAR10_DIR = DATASET_DIR / "cifar-10"
CIFAR100_DIR = DATASET_DIR / "cifar-100"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


# ---------------------------------------------------------------------------
# Tien ich: download / extract / checksum
# ---------------------------------------------------------------------------
def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_session() -> requests.Session:
    """Session co retry/backoff de chiu duoc duong truyen cham/chap chon.
    Dung cho phan TAI THAT (GET), KHONG dung cho HEAD-probe (xem _probe_head)."""
    session = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _probe_head(url: str, timeout: tuple[int, int] = (5, 10)) -> tuple[int, bool]:
    """Kiem tra content-length / Accept-Ranges TRUOC khi tai, bang session
    RIENG khong retry (max_retries=0) va timeout NGAN.

    Ly do tach rieng: neu dung chung session chinh (Retry total=5,
    backoff_factor=1.5, timeout=(15,30)) thi 1 lan HEAD "cham" hoac bi
    redirect toi dich cham co the ngon toi vai PHUT (moi lan thu toi da
    15s connect + 30s read, nhan 5-6 lan thu, cong them backoff) truoc khi
    moi fallback duoc -- day chinh la nguyen nhan script "dung im" rat lau
    ma nguoi dung phai Ctrl+C. Probe nay fail nhanh trong ~10-15s la cung,
    roi fallback ngay ve tai tuan tu (an toan, cham hon nhung khong treo)."""
    probe_session = requests.Session()
    probe_session.mount("https://", HTTPAdapter(max_retries=0))
    probe_session.mount("http://", HTTPAdapter(max_retries=0))
    try:
        head = probe_session.head(url, allow_redirects=True, timeout=timeout)
        total = int(head.headers.get("content-length", 0))
        supports_range = head.headers.get("accept-ranges", "").lower() == "bytes"
        return total, supports_range
    except requests.RequestException as e:
        print(f"[warn] HEAD probe that bai/qua cham ({type(e).__name__}: {e}) "
              f"-> fallback ve tai tuan tu 1 luong (KHONG cho lau hon "
              f"{timeout[0] + timeout[1]}s).")
        return 0, False
    finally:
        probe_session.close()


def _download_segment(session: requests.Session, url: str, dest_path: Path,
                       start: int, end: int, bar: tqdm, lock: threading.Lock,
                       chunk_size: int) -> None:
    headers = {"Range": f"bytes={start}-{end}"}
    with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as resp:
        resp.raise_for_status()
        with open(dest_path, "r+b") as f:
            f.seek(start)
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    with lock:
                        bar.update(len(chunk))


def _download_sequential(session: requests.Session, url: str, dest_path: Path,
                          chunk_size: int) -> None:
    """Tai tuan tu mot luong duy nhat (dung khi server khong ho tro Range)."""
    with session.get(url, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"Downloading {dest_path.name}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_file(url: str, dest_path: Path, expected_md5: str,
                   num_connections: int = 8, chunk_size: int = 1 << 16) -> None:
    """Tai file ve dest_path. Mac dinh tai song song nhieu luong (Range request)
    de tang toc khi server gioi han bang thong tren tung ket noi; tu dong fallback
    ve tai tuan tu neu server khong ho tro Range hoac chi dung 1 luong."""
    if dest_path.exists():
        if md5_of_file(dest_path) == expected_md5:
            print(f"[skip] {dest_path.name} da ton tai va dung checksum.")
            return
        print(f"[warn] {dest_path.name} ton tai nhung sai checksum, tai lai.")
        dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    session = _make_session()

    total, supports_range = _probe_head(url)

    if num_connections <= 1 or not supports_range or total == 0:
        _download_sequential(session, url, dest_path, chunk_size)
    else:
        # cap phat truoc dung dung tong dung luong de cac luong ghi vao dung vi tri
        with open(dest_path, "wb") as f:
            f.truncate(total)

        segment_size = total // num_connections
        ranges = []
        for i in range(num_connections):
            start = i * segment_size
            end = total - 1 if i == num_connections - 1 else start + segment_size - 1
            ranges.append((start, end))

        lock = threading.Lock()
        with tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"Downloading {dest_path.name} ({num_connections} luong)",
        ) as bar:
            with ThreadPoolExecutor(max_workers=num_connections) as executor:
                futures = [
                    executor.submit(_download_segment, session, url, dest_path,
                                     start, end, bar, lock, chunk_size)
                    for start, end in ranges
                ]
                for fut in as_completed(futures):
                    fut.result()  # raise lai loi neu co luong nao fail

    actual_md5 = md5_of_file(dest_path)
    if actual_md5 != expected_md5:
        print(f"[warn] checksum khong khop cho {dest_path.name} "
              f"(expected {expected_md5}, got {actual_md5}). File van duoc giu lai, "
              f"kiem tra lai duong truyen mang neu giai nen loi.")


def extract_archive(archive_path: Path, extract_to: Path) -> Path:
    """Giai nen .tar.gz, hien thi tien trinh bang tqdm. Tra ve thu muc goc sau giai nen."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        top_level = members[0].name.split("/")[0]
        target_root = extract_to / top_level
        if target_root.exists():
            print(f"[skip] {top_level} da duoc giai nen.")
            return target_root
        for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
            tar.extract(member, path=extract_to)
    return target_root


def load_pickle_batch(file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def save_npy(out_dir: Path, split: str, images: np.ndarray, labels: np.ndarray,
             extra: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_images.npy", images)
    np.save(out_dir / f"{split}_labels.npy", labels)
    if extra:
        for name, arr in extra.items():
            np.save(out_dir / f"{split}_{name}.npy", arr)
    print(f"[saved] {split}: images {images.shape}, labels {labels.shape} -> {out_dir}")


def _flat_to_hwc(flat: np.ndarray) -> np.ndarray:
    """(N, 3072) -> (N, 32, 32, 3) uint8."""
    return flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------
def process_cifar10(extracted_dir: Path, out_dir: Path) -> None:
    train_files = [f"data_batch_{i}" for i in range(1, 6)]
    train_images, train_labels = [], []
    for fname in tqdm(train_files, desc="CIFAR-10 train batches"):
        batch = load_pickle_batch(extracted_dir / fname)
        train_images.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    train_images = _flat_to_hwc(np.concatenate(train_images, axis=0))
    train_labels = np.array(train_labels, dtype=np.int64)

    test_batch = load_pickle_batch(extracted_dir / "test_batch")
    test_images = _flat_to_hwc(test_batch[b"data"])
    test_labels = np.array(test_batch[b"labels"], dtype=np.int64)

    meta = load_pickle_batch(extracted_dir / "batches.meta")
    class_names = [c.decode("utf-8") for c in meta[b"label_names"]]
    (out_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    save_npy(out_dir, "train", train_images, train_labels)
    save_npy(out_dir, "test", test_images, test_labels)


# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------
def process_cifar100(extracted_dir: Path, out_dir: Path) -> None:
    for split, fname in tqdm([("train", "train"), ("test", "test")], desc="CIFAR-100 splits"):
        batch = load_pickle_batch(extracted_dir / fname)
        images = _flat_to_hwc(batch[b"data"])
        fine_labels = np.array(batch[b"fine_labels"], dtype=np.int64)
        coarse_labels = np.array(batch[b"coarse_labels"], dtype=np.int64)
        save_npy(out_dir, split, images, fine_labels, extra={"coarse_labels": coarse_labels})

    meta = load_pickle_batch(extracted_dir / "meta")
    fine_names = [c.decode("utf-8") for c in meta[b"fine_label_names"]]
    coarse_names = [c.decode("utf-8") for c in meta[b"coarse_label_names"]]
    (out_dir / "fine_classes.txt").write_text("\n".join(fine_names), encoding="utf-8")
    (out_dir / "coarse_classes.txt").write_text("\n".join(coarse_names), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download va chuan bi CIFAR-10 / CIFAR-100")
    parser.add_argument("--which", choices=["cifar10", "cifar100", "both"], default="both")
    parser.add_argument("--keep-raw", action="store_true",
                         help="Giu lai file .tar.gz va thu muc giai nen trong dataset/_raw")
    parser.add_argument("--connections", type=int, default=8,
                         help="So luong ket noi song song khi tai file (mac dinh 8, "
                              "dat = 1 de tai tuan tu)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.which in ("cifar10", "both"):
        targets.append(("cifar10", CIFAR10_URL, CIFAR10_MD5, CIFAR10_DIR, process_cifar10))
    if args.which in ("cifar100", "both"):
        targets.append(("cifar100", CIFAR100_URL, CIFAR100_MD5, CIFAR100_DIR, process_cifar100))

    for name, url, md5, out_dir, process_fn in targets:
        print(f"\n=== {name.upper()} ===")
        archive_path = RAW_DIR / Path(url).name
        download_file(url, archive_path, md5, num_connections=args.connections)
        extracted_root = extract_archive(archive_path, RAW_DIR)
        process_fn(extracted_root, out_dir)

        if not args.keep_raw:
            shutil.rmtree(extracted_root, ignore_errors=True)

    print("\nXong. Du lieu da duoc luu trong thu muc dataset/.")


if __name__ == "__main__":
    main()