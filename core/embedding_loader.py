"""
embedding_loader.py
--------------------
Loads all real-world embedding datasets from manually downloaded files.
All datasets are expected under:
    /mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main/data/raw/

Datasets supported:
  1. GloVe        — 50d, 100d, 200d, 300d  (glove.6B.*.txt, glove.840B.300d.txt)
  2. Word2Vec     — 300d                    (GoogleNews-vectors-negative300.bin)
  3. BERT         — 768d                    (sentences.txt → GPU encoding)
  4. CIFAR-10 raw — 3072d                   (cifar-10 batch files)
  5. CIFAR-10 ResNet features — 2048d, 512d (extracted via pretrained ResNet-50)
  6. MNIST raw    — 784d                    (IDX binary files)
  7. MNIST PCA    — 50d, 100d, 200d         (derived from raw)
  8. scRNA PBMC   — variable               (10x Genomics MTX format)

All processed arrays are cached as .npy files in data/processed/
to avoid repeated expensive computation.
"""

import os
import sys
import struct
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Project and data paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/data2/naeem/Geometry-and-Concentration-in-High-Dimensions-main")
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU detection ──────────────────────────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
except ImportError:
    CUDA_AVAILABLE = False


# ==============================================================================
# 1. GloVe
# ==============================================================================

def load_glove(dim: int = 300, n_words: int = 50_000, seed: int = 42) -> np.ndarray:
    """
    Load GloVe word vectors from local .txt file.

    Parameters
    ----------
    dim     : 50 | 100 | 200 | 300
    n_words : number of vectors to sample
    seed    : random seed for sampling

    Returns
    -------
    np.ndarray shape (n_words, dim) float32
    """
    cache = PROC_DIR / f"glove_{dim}d_{n_words}.npy"
    if cache.exists():
        logger.info("GloVe %dd: loading from cache %s", dim, cache)
        return np.load(cache)

    # Choose source file
    if dim == 300:
        src = RAW_DIR / "glove" / "glove.840B.300d.txt"
    else:
        src = RAW_DIR / "glove" / f"glove.6B.{dim}d.txt"

    if not src.exists():
        raise FileNotFoundError(
            f"GloVe file not found: {src}\n"
            f"Download from https://nlp.stanford.edu/data/glove.6B.zip (50/100/200d) "
            f"or https://nlp.stanford.edu/data/glove.840B.300d.zip (300d)\n"
            f"Place .txt files in {RAW_DIR / 'glove'}"
        )

    logger.info("GloVe %dd: reading %s …", dim, src)
    vectors = []
    with open(src, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            try:
                vectors.append(np.array(parts[1:], dtype=np.float32))
            except ValueError:
                continue

    vectors = np.stack(vectors, axis=0)
    logger.info("  Loaded %d vectors of dim %d", len(vectors), dim)

    # Subsample
    rng = np.random.default_rng(seed)
    if n_words < len(vectors):
        idx = rng.choice(len(vectors), size=n_words, replace=False)
        vectors = vectors[idx]

    np.save(cache, vectors)
    logger.info("  Cached to %s  shape=%s", cache, vectors.shape)
    return vectors


# ==============================================================================
# 2. Word2Vec
# ==============================================================================

def load_word2vec(n_words: int = 50_000, seed: int = 42) -> np.ndarray:
    """
    Load Google News Word2Vec vectors from binary .bin file.

    Returns
    -------
    np.ndarray shape (n_words, 300) float32
    """
    cache = PROC_DIR / f"word2vec_300d_{n_words}.npy"
    if cache.exists():
        logger.info("Word2Vec: loading from cache %s", cache)
        return np.load(cache)

    src = RAW_DIR / "word2vec" / "GoogleNews-vectors-negative300.bin"
    if not src.exists():
        raise FileNotFoundError(
            f"Word2Vec file not found: {src}\n"
            f"Download GoogleNews-vectors-negative300.bin.gz, gunzip it, "
            f"and place in {RAW_DIR / 'word2vec'}"
        )

    logger.info("Word2Vec: reading binary file %s …", src)
    vectors = []
    with open(src, "rb") as fh:
        # Read header: "num_words dim\n"
        header = b""
        while True:
            ch = fh.read(1)
            if ch == b"\n":
                break
            header += ch
        vocab_size, dim = map(int, header.split())
        logger.info("  Vocab=%d  dim=%d", vocab_size, dim)

        rng = np.random.default_rng(seed)
        # Decide which indices to keep upfront for memory efficiency
        keep = set(rng.choice(vocab_size, size=min(n_words, vocab_size),
                               replace=False).tolist())

        for i in range(vocab_size):
            # Read word (terminated by space)
            word = b""
            while True:
                ch = fh.read(1)
                if ch in (b" ", b"\t"):
                    break
                word += ch
            # Read vector (dim float32 values)
            raw = fh.read(dim * 4)
            if i in keep:
                vec = np.frombuffer(raw, dtype=np.float32).copy()
                vectors.append(vec)
            # Skip newline between entries if present
            nxt = fh.read(1)
            if nxt != b"\n":
                fh.seek(-1, 1)

    vectors = np.stack(vectors[:n_words], axis=0)
    np.save(cache, vectors)
    logger.info("Word2Vec cached: %s  shape=%s", cache, vectors.shape)
    return vectors


# ==============================================================================
# 3. BERT sentence embeddings
# ==============================================================================

def load_bert(n_sentences: int = 50_000,
              model_name: str = "bert-base-uncased",
              seed: int = 42) -> np.ndarray:
    """
    Generate BERT sentence embeddings from sentences.txt using GPU.

    Reads /data/raw/bert_text/sentences.txt (one sentence per line).
    Uses mean-pooling of the final hidden layer.

    Returns
    -------
    np.ndarray shape (n_sentences, 768) float32
    """
    cache = PROC_DIR / f"bert_768d_{n_sentences}.npy"
    if cache.exists():
        logger.info("BERT: loading from cache %s", cache)
        return np.load(cache)

    txt_path = RAW_DIR / "bert_text" / "sentences.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"BERT text file not found: {txt_path}\n"
            f"Create a plain text file with one sentence per line (50,000 lines).\n"
            f"Any large English text corpus works (Wikipedia, BookCorpus, etc.)."
        )

    # Read sentences
    with open(txt_path, encoding="utf-8", errors="ignore") as fh:
        sentences = [ln.strip() for ln in fh if ln.strip()]

    rng = np.random.default_rng(seed)
    if len(sentences) > n_sentences:
        idx = rng.choice(len(sentences), size=n_sentences, replace=False)
        sentences = [sentences[i] for i in sorted(idx)]
    logger.info("BERT: encoding %d sentences with %s …", len(sentences), model_name)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        raise ImportError("Install: pip install transformers torch")

    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    logger.info("  Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()

    batch_size = 256
    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            output = model(**encoded)
            # Mean-pool over token dimension (masked)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            token_embeddings = output.last_hidden_state
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_pooled = (summed / counts).cpu().numpy().astype(np.float32)
            all_embeddings.append(mean_pooled)

            if (start // batch_size) % 20 == 0:
                logger.info("  … %d / %d", start + len(batch), len(sentences))

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(cache, embeddings)
    logger.info("BERT cached: %s  shape=%s", cache, embeddings.shape)
    return embeddings


# ==============================================================================
# 4. CIFAR-10 raw pixels
# ==============================================================================

def _unpickle_cifar(file_path: Path) -> dict:
    import pickle
    with open(file_path, "rb") as fh:
        return pickle.load(fh, encoding="bytes")


def load_cifar10_raw(n_samples: int = 60_000, seed: int = 42) -> np.ndarray:
    """
    Load CIFAR-10 raw pixel data as float32 vectors (3072d).

    Returns
    -------
    np.ndarray shape (n_samples, 3072) float32, values in [0, 1]
    """
    cache = PROC_DIR / f"cifar10_raw_{n_samples}.npy"
    if cache.exists():
        logger.info("CIFAR-10 raw: loading from cache %s", cache)
        return np.load(cache)

    cifar_dir = RAW_DIR / "cifar10"
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    all_data = []

    for bf in batch_files:
        fpath = cifar_dir / bf
        if not fpath.exists():
            raise FileNotFoundError(
                f"CIFAR-10 batch file not found: {fpath}\n"
                f"Download cifar-10-python.tar.gz from "
                f"https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n"
                f"Extract and place batch files in {cifar_dir}"
            )
        d = _unpickle_cifar(fpath)
        all_data.append(d[b"data"])

    data = np.concatenate(all_data, axis=0).astype(np.float32) / 255.0
    logger.info("CIFAR-10 raw: loaded %d images (3072d)", len(data))

    rng = np.random.default_rng(seed)
    if n_samples < len(data):
        idx = rng.choice(len(data), size=n_samples, replace=False)
        data = data[idx]

    np.save(cache, data)
    logger.info("CIFAR-10 raw cached: %s  shape=%s", cache, data.shape)
    return data


# ==============================================================================
# 5. CIFAR-10 ResNet features
# ==============================================================================

def load_cifar10_resnet(layer: str = "layer4",
                         n_samples: int = 60_000,
                         seed: int = 42) -> np.ndarray:
    """
    Extract CIFAR-10 features using pretrained ResNet-50.

    Parameters
    ----------
    layer : 'layer3' (512d intermediate) or 'layer4' / 'avgpool' (2048d)

    Returns
    -------
    np.ndarray shape (n_samples, d) float32
    """
    dim_label = "512d" if layer == "layer3" else "2048d"
    cache = PROC_DIR / f"cifar10_resnet_{dim_label}_{n_samples}.npy"
    if cache.exists():
        logger.info("CIFAR-10 ResNet %s: loading from cache %s", dim_label, cache)
        return np.load(cache)

    try:
        import torch
        import torchvision
        import torchvision.transforms as T
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        raise ImportError("Install: pip install torch torchvision")

    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    logger.info("CIFAR-10 ResNet: extracting features on %s …", device)

    # Build dataset from raw batch files
    cifar_dir = RAW_DIR / "cifar10"
    batch_files = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    images_list, labels_list = [], []
    for bf in batch_files:
        d = _unpickle_cifar(cifar_dir / bf)
        images_list.append(d[b"data"])
        labels_list.extend(d[b"labels"])

    raw_images = np.concatenate(images_list, axis=0)  # (N, 3072) uint8
    # Reshape to (N, 3, 32, 32)
    raw_images = raw_images.reshape(-1, 3, 32, 32)

    # Subsample
    rng = np.random.default_rng(seed)
    n_total = len(raw_images)
    if n_samples < n_total:
        idx = rng.choice(n_total, size=n_samples, replace=False)
        idx.sort()
        raw_images = raw_images[idx]

    # ImageNet normalisation + resize for ResNet
    transform = T.Compose([
        T.Resize(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Simple dataset wrapper
    class CIFARTensor(torch.utils.data.Dataset):
        def __init__(self, imgs):
            self.imgs = torch.from_numpy(imgs.astype(np.float32) / 255.0)

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            return transform(self.imgs[i])

    loader = DataLoader(CIFARTensor(raw_images), batch_size=256,
                        shuffle=False, num_workers=4, pin_memory=CUDA_AVAILABLE)

    # Build ResNet-50 feature extractor
    resnet = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    ).to(device).eval()

    features_list = []
    hook_output = {}

    if layer == "layer3":
        # Hook into layer3 → global average pool → 512d
        def hook_fn(module, input, output):
            hook_output["feat"] = output

        handle = resnet.layer3.register_forward_hook(hook_fn)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                resnet(batch)
                feat = hook_output["feat"]          # (B, 256, H, W) for layer3
                feat = feat.mean(dim=[2, 3])        # global avg pool → (B, 256)
                features_list.append(feat.cpu().numpy())
        handle.remove()
    else:
        # Use penultimate avgpool → 2048d
        extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).eval().to(device)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                feat = extractor(batch).squeeze(-1).squeeze(-1)  # (B, 2048)
                features_list.append(feat.cpu().numpy())

    features = np.concatenate(features_list, axis=0).astype(np.float32)
    np.save(cache, features)
    logger.info("CIFAR-10 ResNet %s cached: %s  shape=%s", dim_label, cache, features.shape)
    return features


# ==============================================================================
# 6. MNIST raw pixels
# ==============================================================================

def _read_idx(path: Path) -> np.ndarray:
    """Read IDX binary format file."""
    with open(path, "rb") as fh:
        magic = struct.unpack(">I", fh.read(4))[0]
        n_dims = magic & 0xFF
        dims = [struct.unpack(">I", fh.read(4))[0] for _ in range(n_dims)]
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return data.reshape(dims)


def load_mnist_raw(n_samples: int = 70_000, seed: int = 42) -> np.ndarray:
    """
    Load MNIST raw pixels as float32 vectors (784d).

    Returns
    -------
    np.ndarray shape (n_samples, 784) float32, values in [0, 1]
    """
    cache = PROC_DIR / f"mnist_raw_784d_{n_samples}.npy"
    if cache.exists():
        logger.info("MNIST raw: loading from cache %s", cache)
        return np.load(cache)

    mnist_dir = RAW_DIR / "mnist"
    train_imgs = mnist_dir / "train-images-idx3-ubyte"
    test_imgs  = mnist_dir / "t10k-images-idx3-ubyte"

    for p in [train_imgs, test_imgs]:
        if not p.exists():
            raise FileNotFoundError(
                f"MNIST file not found: {p}\n"
                f"Download from http://yann.lecun.com/exdb/mnist/\n"
                f"Gunzip and place IDX files in {mnist_dir}"
            )

    train = _read_idx(train_imgs).reshape(-1, 784).astype(np.float32) / 255.0
    test  = _read_idx(test_imgs).reshape(-1, 784).astype(np.float32) / 255.0
    data  = np.concatenate([train, test], axis=0)
    logger.info("MNIST raw: loaded %d images (784d)", len(data))

    rng = np.random.default_rng(seed)
    if n_samples < len(data):
        idx = rng.choice(len(data), size=n_samples, replace=False)
        data = data[idx]

    np.save(cache, data)
    logger.info("MNIST raw cached: %s  shape=%s", cache, data.shape)
    return data


def load_mnist_pca(dim: int = 50, n_samples: int = 70_000, seed: int = 42) -> np.ndarray:
    """
    Load MNIST projected to `dim` PCA components.

    Parameters
    ----------
    dim : 50 | 100 | 200

    Returns
    -------
    np.ndarray shape (n_samples, dim) float32
    """
    cache = PROC_DIR / f"mnist_pca_{dim}d_{n_samples}.npy"
    if cache.exists():
        logger.info("MNIST PCA %dd: loading from cache %s", dim, cache)
        return np.load(cache)

    logger.info("MNIST PCA %dd: computing from raw …", dim)
    raw = load_mnist_raw(n_samples=n_samples, seed=seed)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=seed)
    projected = pca.fit_transform(raw).astype(np.float32)
    explained = pca.explained_variance_ratio_.sum()
    logger.info("  PCA %dd explains %.1f%% variance", dim, 100 * explained)

    np.save(cache, projected)
    logger.info("MNIST PCA %dd cached: %s  shape=%s", dim, cache, projected.shape)
    return projected


# ==============================================================================
# 7. scRNA PBMC 3k
# ==============================================================================

def load_scrna(n_cells: int = 3_000, n_genes: int = 2_000, seed: int = 42) -> np.ndarray:
    """
    Load and preprocess 10x Genomics PBMC 3k scRNA-seq data.

    Preprocessing steps:
      1. Load sparse matrix
      2. Filter cells (min 200 genes) and genes (min 3 cells)
      3. Log-normalise: log1p(counts / total * 1e4)
      4. Select top `n_genes` highly variable genes

    Returns
    -------
    np.ndarray shape (n_cells, n_genes) float32
    """
    cache = PROC_DIR / f"scrna_pbmc_{n_cells}cells_{n_genes}genes.npy"
    if cache.exists():
        logger.info("scRNA PBMC: loading from cache %s", cache)
        return np.load(cache)

    scrna_dir = RAW_DIR / "scrna"
    mtx_file  = scrna_dir / "matrix.mtx.gz"

    if not mtx_file.exists():
        raise FileNotFoundError(
            f"scRNA MTX file not found: {mtx_file}\n"
            f"Download 'Feature / cell matrix (filtered)' from:\n"
            f"https://www.10xgenomics.com/datasets/pbmc-3-k-pbmcs-from-a-healthy-donor-1-standard-1-1-0\n"
            f"Place barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz in {scrna_dir}"
        )

    try:
        import scanpy as sc
        import anndata
    except ImportError:
        raise ImportError("Install: pip install scanpy anndata")

    logger.info("scRNA PBMC: loading with scanpy …")
    adata = sc.read_10x_mtx(str(scrna_dir), var_names="gene_symbols", cache=False)
    logger.info("  Raw shape: %s", adata.shape)

    # Standard preprocessing pipeline
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_genes)
    adata = adata[:, adata.var.highly_variable]
    logger.info("  After preprocessing: %s", adata.shape)

    # Extract dense matrix
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)

    # Subsample cells
    rng = np.random.default_rng(seed)
    if n_cells < X.shape[0]:
        idx = rng.choice(X.shape[0], size=n_cells, replace=False)
        X = X[idx]

    np.save(cache, X)
    logger.info("scRNA PBMC cached: %s  shape=%s", cache, X.shape)
    return X


# ==============================================================================
# Unified registry
# ==============================================================================

DATASET_REGISTRY = {
    # name            : (loader_fn,              default kwargs,               ambient_dim)
    "glove_50d"       : (load_glove,             {"dim": 50,  "n_words": 50_000}, 50),
    "glove_100d"      : (load_glove,             {"dim": 100, "n_words": 50_000}, 100),
    "glove_200d"      : (load_glove,             {"dim": 200, "n_words": 50_000}, 200),
    "glove_300d"      : (load_glove,             {"dim": 300, "n_words": 50_000}, 300),
    "word2vec_300d"   : (load_word2vec,          {"n_words": 50_000},             300),
    "bert_768d"       : (load_bert,              {"n_sentences": 50_000},         768),
    "cifar10_raw"     : (load_cifar10_raw,       {"n_samples": 60_000},           3072),
    "cifar10_resnet2048": (load_cifar10_resnet,  {"layer": "layer4", "n_samples": 60_000}, 2048),
    "cifar10_resnet512": (load_cifar10_resnet,   {"layer": "layer3", "n_samples": 60_000}, 512),
    "mnist_raw"       : (load_mnist_raw,         {"n_samples": 70_000},           784),
    "mnist_pca50"     : (load_mnist_pca,         {"dim": 50,  "n_samples": 70_000}, 50),
    "mnist_pca100"    : (load_mnist_pca,         {"dim": 100, "n_samples": 70_000}, 100),
    "mnist_pca200"    : (load_mnist_pca,         {"dim": 200, "n_samples": 70_000}, 200),
    "scrna_pbmc"      : (load_scrna,             {"n_cells": 3_000, "n_genes": 2_000}, 2000),
}


def load_dataset(name: str, seed: int = 42) -> np.ndarray:
    """
    Load a dataset by registry name.

    Parameters
    ----------
    name : str  — key in DATASET_REGISTRY
    seed : int

    Returns
    -------
    np.ndarray float32
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )
    loader_fn, kwargs, _ = DATASET_REGISTRY[name]
    return loader_fn(**kwargs, seed=seed)


def get_ambient_dim(name: str) -> int:
    """Return ambient dimension for a registered dataset."""
    return DATASET_REGISTRY[name][2]
