# AquaSign: Underwater Hand-Gesture Recognition

Underwater hand-gesture recognition trained from scratch, evaluated across pool and sea domains with unsupervised domain adaptation and abstention.

---

## 1. Quick Start

### Environment Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Unit Tests
```bash
PYTHONPATH=. pytest -v tests/
```

---

## 2. Dataset Pipeline (Phase 0 & Foundation)

1. **Phase 0 Inspection & Stop Condition Check:**
   ```bash
   PYTHONPATH=. python -m aquasign.data.inspect_dataset
   ```
   Generates `DATASET_REPORT.md` and montages in `results/figs/sample_grid_pool.png` and `sample_grid_sea.png`.

2. **Index Generation & Class Filtering:**
   ```bash
   PYTHONPATH=. python -m aquasign.data.build_index
   ```
   Filters classes with $\ge 40$ samples in both domains and generates `data/index.csv` and `classes.json`.

3. **Leakage-Safe Block Splits:**
   ```bash
   PYTHONPATH=. python -m aquasign.data.splits
   ```
   Generates contiguous 50-frame block partitions into `data/splits.json` ensuring 0% sample or pair leakage.

4. **Cache Generation:**
   ```bash
   PYTHONPATH=. python -m aquasign.data.cache
   ```
   Pre-resizes images to $(144 \times 192)$ and saves uint8 arrays to `cache/pool_144x192.npy` and `cache/sea_144x192.npy`.

---

## 3. Training & Evaluation

### Stage 1 Classifier Training
```bash
# Baseline (No correction, No alignment)
PYTHONPATH=. python -m aquasign.train --config configs/pool2sea.yaml --corr 0 --align 0 --seed 0

# Full Novelty Model (With correction and domain alignment)
PYTHONPATH=. python -m aquasign.train --config configs/pool2sea.yaml --corr 1 --align 1 --seed 0
```

### Evaluation (with & without AdaBN)
```bash
# Standard evaluation
PYTHONPATH=. python -m aquasign.evaluate --checkpoint results/runs/pool2sea_c0a0_s0/best.pt

# With Test-Time AdaBN adaptation
PYTHONPATH=. python -m aquasign.evaluate --checkpoint results/runs/pool2sea_c0a0_s0/best.pt --adabn
```
Metrics are output directly to terminal and saved to `eval_adabn_off.json` / `eval_adabn_on.json`.
