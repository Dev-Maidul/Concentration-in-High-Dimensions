# Geometry and Concentration in High Dimensions

**Empirical Scaling Laws in High-Dimensional Geometry**

A complete research experiment framework for analyzing concentration phenomena in high-dimensional spaces, including synthetic experiments, universality analysis, and real embedding validation.

---

## Overview

This codebase implements large-scale computational experiments to empirically study:

1. Norm Concentration — how vector norms concentrate in high dimensions
2. Distance Geometry — pairwise distance behavior and nearest neighbor phenomena
3. Hubness — the emergence of hub points in high-dimensional kNN graphs
4. Johnson-Lindenstrauss Projections — random projection methods and distortion analysis
5. Random Matrix Theory — eigenvalue distributions and Marchenko-Pastur comparisons
6. Cross-Phenomenon Scaling Laws — unified analysis of concentration rates
7. Universality of the -0.623 Exponent — Section 5 experiments and heuristic derivation
8. Real Embedding Validation — Section 6 experiments on GloVe, Word2Vec, BERT, CIFAR-10, MNIST, scRNA

---

## Project Structure

```
Geometry-and-Concentration-in-High-Dimensions-main/
│
├── config/
│   ├── global_config.py               Global configuration singleton
│   └── experiment_config.yaml         YAML configuration reference
│
├── core/                              Core library modules
│   ├── random_generators.py           Synthetic distribution generators
│   ├── metrics.py                     GPU-accelerated metric computation
│   ├── projection_methods.py          JL projection methods (Gaussian, sparse, structured)
│   ├── spectral_methods.py            Eigenvalue analysis and Marchenko-Pastur
│   ├── scaling_analysis.py            Power-law fitting and comparison
│   ├── plotting_utils.py              Shared plotting utilities
│   ├── reproducibility.py             Logging, caching, result management
│   ├── embedding_loader.py            NEW: loads all 14 real dataset variants
│   ├── intrinsic_dim.py               NEW: TwoNN, MLE, PCA, correlation dim estimators
│   └── dependence_analysis.py         NEW: -0.623 heuristic derivation machinery
│
├── experiments/
│   ├── norm_concentration/
│   │   └── run_norm_experiment.py     Section 4, Law 1
│   ├── distance_geometry/
│   │   ├── run_distance_experiment.py Section 4, Law 2
│   │   └── hubness_analysis.py        Section 4, Law 3
│   ├── jl_projections/
│   │   └── run_jl_experiment.py       Section 4, Law 4
│   ├── spectral_analysis/
│   │   └── run_spectral_experiment.py Section 4, Laws 5-6
│   ├── cross_analysis/
│   │   └── run_scaling_comparison.py  Section 7 scaling hierarchy
│   ├── universality/
│   │   └── run_universality_experiment.py  NEW: Section 5 (all three parts)
│   └── real_embeddings/
│       ├── preprocess_embeddings.py   NEW: one-time setup, run before everything else
│       ├── run_embedding_geometry.py  NEW: Section 6 main experiment
│       └── analyze_discrepancies.py   NEW: Section 6.4 targeted analyses
│
├── visualization/
│   └── generate_all_figures.py        Original Section 4 figures
│
├── tables/
│   ├── generate_all_tables.py         Original Section 4 tables
│   └── generate_prediction_table.py   NEW: Section 7 prediction table and figures
│
├── data/
│   ├── raw/                           Place downloaded files here (see below)
│   │   ├── glove/
│   │   ├── word2vec/
│   │   ├── bert_text/
│   │   ├── bert_model/
│   │   ├── cifar10/
│   │   ├── mnist/
│   │   └── scrna/
│   └── processed/                     Auto-generated .npy cache files (do not edit)
│
├── results/
│   ├── raw/                           Raw experimental outputs
│   ├── processed/                     Processed intermediate results
│   ├── figures/                       Publication figures (PDF and PNG)
│   └── tables/                        Publication tables (LaTeX and CSV)
│
├── main.py                            Main experiment runner (updated)
├── generate_all_outputs.py            Original full pipeline runner
└── requirements.txt                   Python dependencies
```

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (Titan V or equivalent recommended for BERT and ResNet extraction)
- 16+ GB RAM recommended
- Packages: see requirements.txt, plus `anndata` and `scanpy`

### Setup

```bash
cd Geometry-and-Concentration-in-High-Dimensions-main
pip install -r requirements.txt
pip install anndata scanpy --break-system-packages
```

---

## Dataset Setup

All datasets must be downloaded manually and placed in the correct locations. The code reads local files only — no automatic downloads except ResNet-50 weights and BERT model files (see below).

### Required file locations

```
data/raw/glove/
    glove.6B.50d.txt
    glove.6B.100d.txt
    glove.6B.200d.txt
    glove.840B.300d.txt

data/raw/word2vec/
    GoogleNews-vectors-negative300.bin      (gunzip the .bin.gz first)

data/raw/bert_text/
    sentences.txt                           (one sentence per line, 50000 lines)

data/raw/bert_model/
    config.json
    tokenizer_config.json
    tokenizer.json
    vocab.txt
    model.safetensors

data/raw/cifar10/
    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    test_batch

data/raw/mnist/
    train-images-idx3-ubyte                 (gunzip first)
    train-labels-idx1-ubyte                 (gunzip first)
    t10k-images-idx3-ubyte                  (gunzip first)
    t10k-labels-idx1-ubyte                  (gunzip first)

data/raw/scrna/
    barcodes.tsv.gz
    features.tsv.gz
    matrix.mtx.gz
```

### Download sources

- GloVe: https://nlp.stanford.edu/data/glove.6B.zip and https://nlp.stanford.edu/data/glove.840B.300d.zip
- Word2Vec: Google News vectors (GoogleNews-vectors-negative300.bin.gz)
- BERT model files: https://huggingface.co/bert-base-uncased/resolve/main/ (download config.json, tokenizer_config.json, tokenizer.json, vocab.txt, model.safetensors)
- ResNet-50 weights: https://download.pytorch.org/models/resnet50-11ad3fa6.pth — place at /home/username/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- MNIST: http://yann.lecun.com/exdb/mnist/
- scRNA PBMC 3k: https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz

### Generating sentences.txt for BERT

```python
import nltk
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize
import random

sentences = []
for fileid in gutenberg.fileids():
    text = gutenberg.raw(fileid)
    sentences.extend(sent_tokenize(text))

sentences = [s.strip().replace('\n', ' ') for s in sentences if len(s.split()) >= 5]
random.seed(42)
random.shuffle(sentences)

with open('data/raw/bert_text/sentences.txt', 'w') as f:
    for s in sentences[:50000]:
        f.write(s + '\n')
```

---

## Running Experiments

### Correct running order

The preprocessing step must run before embedding geometry. Everything else is independent.

```bash
# Step 1 — one-time preprocessing of all real datasets (~30-60 min on GPU)
python main.py --experiment preprocess_embeddings

# Step 2 — Section 4: original synthetic experiments
python main.py --experiment norm_concentration
python main.py --experiment distance_geometry
python main.py --experiment hubness
python main.py --experiment jl_projections
python main.py --experiment spectral
python main.py --experiment scaling_comparison

# Step 3 — Section 5: universality of -0.623 exponent (~2 hr)
python main.py --experiment universality

# Step 4 — Section 6: real embedding geometry (~1 hr)
python main.py --experiment embedding_geometry

# Step 5 — Section 6.4: discrepancy analysis (~5 min)
python main.py --experiment discrepancy

# Step 6 — Section 7: prediction table and figures (~2 min)
python main.py --experiment prediction_table
```

### Run everything at once

```bash
python main.py --experiment preprocess_embeddings 2>&1 | tee preprocess.log && \
python main.py --experiment universality 2>&1 | tee universality.log && \
python main.py --experiment embedding_geometry 2>&1 | tee embedding.log && \
python main.py --experiment discrepancy 2>&1 | tee discrepancy.log && \
python main.py --experiment prediction_table 2>&1 | tee prediction.log
```

### Fast versions for testing

```bash
python main.py --experiment universality_fast        # 10 trials instead of 50
python main.py --experiment embedding_geometry_fast  # 2 trials instead of 5
```

### List all available experiments

```bash
python main.py --list
```

### Run original pipeline only

```bash
python generate_all_outputs.py
```

---

## Configuration

Edit `config/global_config.py` to change:

```python
RANDOM_SEED = 42
DIMENSIONS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
FIXED_SAMPLE_SIZE = 10000
N_TRIALS = 50
DISTRIBUTIONS = ['gaussian', 'uniform', 'laplace', 'student_t']
N_JOBS = 1
```

---

## Output Files

### Figures (results/figures/)

Norm Concentration:
- NC1_norm_histogram_*.png/pdf
- NC2_mean_norm_vs_dimension.png/pdf
- NC3_relative_variance_scaling.png/pdf
- NC4_shell_thickness.png/pdf

Distance Geometry:
- DG1_distance_histogram_evolution.png/pdf
- DG2_relative_contrast.png/pdf
- DG3_nn_ratio.png/pdf
- DG4_cosine_similarity.png/pdf
- DG5_hubness_*.png/pdf

JL Projections:
- JL2_failure_probability_*.png/pdf

Spectral Analysis:
- SP1_eigenvalue_dist_*.png/pdf
- SP2_spectral_convergence.png/pdf
- SP3_aspect_ratio_study.png/pdf
- SP4_pca_stability_*.png/pdf

Cross-Analysis:
- CP1_unified_scaling_laws.png/pdf

New (Sections 5-7):
- figure_scaling_law_overlay.pdf/png
- figure_hierarchy_updated.pdf/png

### Tables (results/tables/)

Original:
- NC1_powerlaw_fits.csv/.tex
- DG1_hubness_statistics.csv/.tex
- JL1_projection_thresholds.csv/.tex
- SP1_spectral_convergence.csv/.tex
- CP1_scaling_laws.csv/.tex

New:
- table_prediction.tex

### Raw results (results/raw/)

- norm_concentration_results.pkl
- distance_geometry_results.pkl
- hubness_results.pkl
- jl_projection_results.pkl
- spectral_analysis_results.pkl
- universality_results.pkl
- dependence_results.pkl
- exponent_vs_n_results.pkl
- real_embeddings/embedding_geometry_full.json
- real_embeddings/scaling_law_validation.json
- real_embeddings/discrepancy_analysis.json
- real_embeddings/intrinsic_dim.json

---

## Experiment Details

### Original experiments (Sections 3-4)

**Norm Concentration**
Measures how vector norms concentrate as dimension increases. Computes mean norm, standard deviation, relative variance, and shell thickness ratio across 4 distributions and 10 dimensions.

**Distance Geometry**
Analyzes pairwise distance distributions and nearest neighbor behavior. Key output is the relative contrast Crel(d) scaling law with exponent -0.623.

**Hubness Analysis**
Quantifies the hubness phenomenon in kNN graphs via Gini coefficient and skewness. Identifies the d=100 threshold where Gini exceeds 0.75.

**Johnson-Lindenstrauss Projections**
Tests Gaussian, sparse, and structured random projections. Measures distortion and failure probabilities at epsilon in {0.1, 0.2, 0.3}.

**Spectral Analysis**
Compares empirical eigenvalue distributions with Marchenko-Pastur theory. Studies the phase transition at aspect ratio gamma=1.

**Cross-Phenomenon Scaling**
Compares scaling exponents across all phenomena and builds the unified scaling hierarchy.

### New experiments (Sections 5-7)

**Universality (Section 5)**
Three sub-experiments: (1) fits power-law exponents with bootstrap confidence intervals across 10 distributions and performs t-test against the null hypothesis of -0.5; (2) empirically verifies Corr(Dij^2, Dik^2) = 1/4 across dimensions; (3) tests whether the exponent varies with sample size n in the direction predicted by the effective-independence model.

**Real Embedding Preprocessing**
One-time setup that loads all 14 dataset variants, estimates intrinsic dimension using TwoNN and Levina-Bickel MLE, and caches processed arrays. Results in data/processed/intrinsic_dimensions.json.

**Real Embedding Geometry (Section 6)**
Runs the full geometric analysis pipeline (norm, distance, hubness, projections) on all real datasets with multiple subsampling trials for uncertainty estimates. Compares observations to scaling law predictions at estimated d_int.

**Discrepancy Analysis (Section 6.4)**
Three targeted analyses: BERT (does training counteract geometric pathology?), GloVe multi-dimension (does d_int rescaling improve law accuracy?), scRNA (does heavy-tail amplification match Law 6 predictions?).

**Prediction Table (Section 7)**
Generates the unified practitioner prediction table mapping d_int to predicted geometric behavior, with real embedding data points overlaid. Produces LaTeX source and two figures.

---

## Caching Behavior

All expensive computations are cached automatically:

- Processed embedding arrays: data/processed/*.npy
- Intrinsic dimension estimates: data/processed/intrinsic_dimensions.json
- Experiment results: results/raw/*.pkl and results/raw/real_embeddings/*.json

Rerunning any experiment is safe — cached results are loaded instantly. To force recomputation, delete the relevant cache file.

---

## Troubleshooting

**Out of memory during distance computation**
Reduce DISTANCE_SUBSAMPLE in global_config.py from 1000 to 500.

**BERT fails with connection error**
The BERT model files must be downloaded manually and placed in data/raw/bert_model/. The code will not attempt to download them if the folder exists.

**Preprocessing crashes with JSON error**
Delete data/processed/intrinsic_dimensions.json and rerun. A previous crash may have left a corrupted cache file.

**TOKENIZERS_PARALLELISM warning during ResNet extraction**
Harmless. Add os.environ["TOKENIZERS_PARALLELISM"] = "false" at the top of core/embedding_loader.py to suppress it.

**Slow Word2Vec loading**
The binary file is 3.4 GB. First load takes 10-15 minutes. Subsequent runs load from the cached .npy file in under one second.

---

## Key Results

Intrinsic dimension estimates from preprocessing:

| Dataset | Ambient dim | d_int (TwoNN) | d_int (MLE) | Agreement |
|---|---|---|---|---|
| GloVe-50 | 50 | 19.2 | 25.2 | No |
| GloVe-100 | 100 | 26.7 | 37.3 | No |
| GloVe-200 | 200 | 33.9 | 46.7 | No |
| GloVe-300 | 300 | 39.2 | 47.0 | Yes |
| Word2Vec-300 | 300 | 45.3 | 63.8 | No |
| CIFAR-10 raw | 3072 | 27.8 | 29.0 | Yes |

---

## Citation

```bibtex
@article{,
  title={Empirical Scaling Laws in High-Dimensional Geometry},
  author={Islam, MD Maidul},
  journal={},
  year={2026}
}
```
