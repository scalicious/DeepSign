# DATASET REPORT: CADDY Underwater Stereo-Vision Gestures

## Executive Summary & Stop Condition Check
- **Stop Condition Triggered:** **NO (PROCEED)**
- **Surviving Gesture Classes (>= 40 per domain):** **15 classes** (Requirement: >= 8 classes)
- **Surviving Pool Pairs:** **4724** (Requirement: >= 500)
- **Surviving Sea Pairs:** **4467** (Requirement: >= 500)
- **Conclusion:** The cross-domain experiment between Pool and Sea is **VALID** and passes all constraints.

---

## 1. Dataset Provenance & License Verification
- **Official Source:** EU FP7 Project CADDY (`http://caddy-underwater-datasets.ge.issia.cnr.it`)
- **Mirror:** Kaggle `ssahu912/underwater-caddy-gestures`
- **Citation:** A. Gomez Chavez, A. Ranieri, D. Chiarella, E. Zereik, A. Babić, and A. Birk, *"Caddy underwater stereo-vision dataset for human–robot interaction (HRI) in the context of diver activities,"* Journal of Marine Science and Engineering, vol. 7, no. 1, 2019.
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** (free for research, commercial use, modification with attribution).

---

## 2. Directory Structure and Raw Formats
- Root format: Stereo image pairs stored as JPEG (`.jpg`).
- Tree layout (depth 3):
- **CADDY_gestures_complete_v2_release/** (8 subdirs, 2 files)
  - **biograd-C/** (2 subdirs, 0 files)
    - **true_negatives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 2974 files)
    - **true_positives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 3610 files)
  - **genova-A/** (2 subdirs, 0 files)
    - **true_negatives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 5574 files)
    - **true_positives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 6640 files)
  - **biograd-B/** (2 subdirs, 0 files)
    - **true_negatives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 1594 files)
    - **true_positives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 1950 files)
  - **brodarski-C/** (2 subdirs, 0 files)
    - **true_negatives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 424 files)
    - **true_positives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 1546 files)
  - **brodarski-D/** (2 subdirs, 0 files)
    - **true_negatives/** (1 subdirs, 0 files)
      - **raw/** (0 subdirs, 608 files)
    - **true_positives/** (1 subdirs, 0 files)

---

## 3. Annotations and Stereo Pair Identification
- **Format:** CSV (`CADDY_gestures_all_true_positives_release_v2.csv` and `CADDY_gestures_all_true_negatives_release_v2.csv`).
- **Stereo Pair Structure:**
  - Left image: `.../{scenario}_{frame:05d}_left.jpg`
  - Right image: `.../{scenario}_{frame:05d}_right.jpg`
  - Stereo baseline: Captured with Point Grey Bumblebee XB3 stereo system.
  - Model pipeline uses **left image only** as specified; right image is tracked via `pair_id`.
- **ROI / Bounding Boxes:** Yes, hand bounding box annotations exist as `[x, y, w, h]` strings in columns `roi left` and `roi right`.
- **Sample annotation per scenario:**
```json
{
  "biograd-A": {
    "left": "/biograd-A/true_positives/raw/biograd-A_00000_left.jpg",
    "right": "/biograd-A/true_positives/raw/biograd-A_00000_right.jpg",
    "label_name": "num_delimiter",
    "label_id": 10,
    "roi_left": "[237,236,54,65]",
    "roi_right": "[155,236,54,65]"
  },
  "biograd-B": {
    "left": "/biograd-B/true_positives/raw/biograd-B_00000_left.jpg",
    "right": "/biograd-B/true_positives/raw/biograd-B_00000_right.jpg",
    "label_name": "two",
    "label_id": 12,
    "roi_left": "[279,378,63,102]",
    "roi_right": "[173,378,63,102]"
  },
  "biograd-C": {
    "left": "/biograd-C/true_positives/raw/biograd-C_00000_left.jpg",
    "right": "/biograd-C/true_positives/raw/biograd-C_00000_right.jpg",
    "label_name": "backwards",
    "label_id": 5,
    "roi_left": "[420,266,85,85]",
    "roi_right": "[294,266,85,85]"
  },
  "brodarski-A": {
    "left": "/brodarski-A/true_positives/raw/brodarski-A_00000_left.jpg",
    "right": "/brodarski-A/true_positives/raw/brodarski-A_00000_right.jpg",
    "label_name": "two",
    "label_id": 12,
    "roi_left": "[332,175,59,82]",
    "roi_right": "[243,175,59,82]"
  },
  "brodarski-B": {
    "left": "/brodarski-B/true_positives/raw/brodarski-B_00000_left.jpg",
    "right": "/brodarski-B/true_positives/raw/brodarski-B_00000_right.jpg",
    "label_name": "start_comm",
    "label_id": 0,
    "roi_left": "[320,138,57,76]",
    "roi_right": "[234,138,57,76]"
  },
  "brodarski-C": {
    "left": "/brodarski-C/true_positives/raw/brodarski-C_00000_left.jpg",
    "right": "/brodarski-C/true_positives/raw/brodarski-C_00000_right.jpg",
    "label_name": "start_comm",
    "label_id": 0,
    "roi_left": "[262,119,67,112]",
    "roi_right": "[135,119,67,112]"
  },
  "brodarski-D": {
    "left": "/brodarski-D/true_positives/raw/brodarski-D_00000_left.jpg",
    "right": "/brodarski-D/true_positives/raw/brodarski-D_00000_right.jpg",
    "label_name": "start_comm",
    "label_id": 0,
    "roi_left": "[353,80,105,147]",
    "roi_right": "[188,80,105,147]"
  },
  "genova-A": {
    "left": "/genova-A/true_positives/raw/genova-A_00000_left.jpg",
    "right": "/genova-A/true_positives/raw/genova-A_00000_right.jpg",
    "label_name": "photo",
    "label_id": 4,
    "roi_left": "[310,84,209,235]",
    "roi_right": "[37,84,209,235]"
  }
}
```

---

## 4. Image Resolution and Characteristics
- **Resolution:** Uniformly `(640, 480)` across tested samples (Aspect ratio 4:3).
- **Target Model Input:** Resized to `144 x 192` (Aspect ratio 4:3), completely preserving geometric proportions without distortion.
- **Color Mode:** 3-channel RGB.

---

## 5. Temporal Order Verification
- Filename format follows `{scenario}_{index:05d}_left.jpg`.
- Frame indices are monotonically sequential per trial session, preserving strict chronological / temporal sequence.
- **Leakage-Safe Strategy:** Contiguous block partitioning (block size 50) based on `order_key = frame_index` prevents near-duplicate frame leakage between train, val, calib, and test sets.

---

## 6. Domain Mapping (Pool vs Sea)
Sourced directly from the CADDY paper (Chavez et al., JMSE 2019, Section 2.1 & Table 1):
- **Sea Scenarios:** `biograd-A`, `biograd-B`, `biograd-C` (Recorded in open sea conditions off Biograd na Moru, Croatia). Total pairs = 4,508.
- **Pool Scenarios:** `brodarski-A`, `brodarski-B`, `brodarski-C`, `brodarski-D` (Brodarski Institute indoor pool, Zagreb) and `genova-A` (CNR outdoor diving pool, Genoa, Italy). Total pairs = 4,731.

---

## 7. Scenario x Class Distribution (Raw Stereo Pairs)

```
label name   backwards  boat  carry  down  end_comm  five  four  here  mosaic  num_delimiter  one  photo  start_comm  three  two   up
scenario                                                                                                                             
biograd-A           30    26     29    15       412    28    89    64      58            302   29     37         468     50   74   17
biograd-B           11    21     36    13       193    13    33    36      22            183   42     16         249     44   46   17
biograd-C          346    25    194    52       144     0    38    37      24            122    0    313         242     69   34  165
brodarski-A          0    32     35     0        30     0     0    30       0              7    0      0          61      0   32    0
brodarski-B          0     7      6     0        12     0     5     8       5              2    0      0          36      0    6    0
brodarski-C          0    77     48     0       179     0     0    36      37             68    0      0         145    112   71    0
brodarski-D          0    40     20     0        61     0     0    20      28             43    0      0          73     18   21    0
genova-A           174   141    349   382       287     7    70    30      52            265   90    559         546     95  120  153
```

---

## 8. Class Overlap Analysis (Threshold = 40 pairs per domain)

### Retained Classes (15 classes):
| Class | Pool Count | Sea Count | Total Count |
|---|---|---|---|
| `backwards` | 174 | 387 | 561 |
| `boat` | 297 | 72 | 369 |
| `carry` | 458 | 259 | 717 |
| `down` | 382 | 80 | 462 |
| `end_comm` | 569 | 749 | 1318 |
| `four` | 75 | 160 | 235 |
| `here` | 124 | 137 | 261 |
| `mosaic` | 122 | 104 | 226 |
| `num_delimiter` | 385 | 607 | 992 |
| `one` | 90 | 71 | 161 |
| `photo` | 559 | 366 | 925 |
| `start_comm` | 861 | 959 | 1820 |
| `three` | 225 | 163 | 388 |
| `two` | 250 | 154 | 404 |
| `up` | 153 | 199 | 352 |

### Dropped Classes (1 classes failing >= 40 per domain):
| Class | Pool Count | Sea Count | Reason |
|---|---|---|---|
| `five` | 7 | 41 | Under threshold in pool |

### Negative Class (True Negatives / No-Gesture):
- Pool negative pairs: 3593
- Sea negative pairs: 3597
- Both domains easily satisfy threshold for negative class inclusion if enabled (`include_negative_class: true`).

---

## 9. Visual Samples
- Pool sample montage saved at: `results/figs/sample_grid_pool.png`
- Sea sample montage saved at: `results/figs/sample_grid_sea.png`
