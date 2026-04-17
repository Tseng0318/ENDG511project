# Corrosion Detection — Self-Supervised vs Supervised Comparison

Binary classification: **CORROSION** vs **NO CORROSION**  
Dataset: 2,522 train | 653 val | 182 ID test | 850 OOD test | 183 grayscale test  
Backbone: ResNet-18 (Supervised, SimCLR, SupCon, BYOL) · ViT-Base (fine-tuned)

---

## 1. In-Distribution (ID) Test Results

| Method | ID Test Acc | Softmax AUROC | Feature AUROC | Cos (ID) | Cos (OOD) | OOD Reject Rate |
|--------|:-----------:|:-------------:|:-------------:|:--------:|:---------:|:---------------:|
| Supervised | 0.9560 | 0.6852 | 0.8302 | 0.8169 | 0.7147 | 0.6376 |
| SimCLR | 0.8736 | 0.6480 | 0.8391 | 0.7329 | 0.6501 | 0.6788 |
| SupCon | 0.9341 | 0.7745 | 0.8189 | 0.8051 | 0.6970 | 0.5859 |
| BYOL | 0.9341 | 0.6208 | 0.7569 | 0.7070 | 0.6456 | 0.5412 |
| **ViT-Base** | **0.9560** | **0.8413** | 0.8196 | 0.5710 | 0.3437 | 0.4871 |

> **Best ID Accuracy:** Supervised & ViT-Base (tie at 96%)  
> **Best OOD Detection (Feature AUROC):** SimCLR (0.8391)  
> **Best Softmax AUROC:** ViT-Base (0.8413)  
> **Highest OOD Reject Rate:** SimCLR (67.9%)

### Per-Method Classification Reports (ID Test)

<details>
<summary>Supervised</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.97      0.95      0.96        99
 NOCORROSION       0.94      0.96      0.95        83
    accuracy                           0.96       182
```
</details>

<details>
<summary>SimCLR</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.90      0.86      0.88        99
 NOCORROSION       0.84      0.89      0.87        83
    accuracy                           0.87       182
```
</details>

<details>
<summary>SupCon</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.96      0.92      0.94        99
 NOCORROSION       0.91      0.95      0.93        83
    accuracy                           0.93       182
```
</details>

<details>
<summary>BYOL</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.93      0.95      0.94        99
 NOCORROSION       0.94      0.92      0.93        83
    accuracy                           0.93       182
```
</details>

<details>
<summary>ViT-Base</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.99      0.93      0.96        99
 NOCORROSION       0.92      0.99      0.95        83
    accuracy                           0.96       182
```
</details>

---

## 2. OOD Detection Summary

| Method | Feature AUROC | Threshold | ID Coverage | Accepted ID Acc | OOD Reject Rate |
|--------|:-------------:|:---------:|:-----------:|:---------------:|:---------------:|
| Supervised | 0.8302 | 0.7434 | 83.5% | **98.7%** | 63.8% |
| **SimCLR** | **0.8391** | 0.6870 | 85.2% | 87.1% | **67.9%** |
| SupCon | 0.8189 | 0.7138 | 85.7% | 96.8% | 58.6% |
| BYOL | 0.7569 | 0.6506 | 86.8% | 93.7% | 54.1% |
| ViT-Base | 0.8196 | 0.3116 | **90.7%** | 98.8% | 48.7% |

> Feature-space cosine similarity is used for OOD detection.  
> **SimCLR** achieves the best Feature AUROC and highest OOD rejection rate.  
> **ViT-Base** achieves the best softmax AUROC (0.8413) and highest ID coverage with near-perfect accuracy on accepted samples.

---

## 3. Grayscale Domain Shift Robustness

Models trained on RGB images were evaluated on a **grayscale copy** of the test set (183 images).

![Grayscale Accuracy Drop](grayscale_accuracy_drop.png)

### Accuracy: RGB vs Grayscale

| Method | RGB Acc | Gray Acc | Acc Drop |
|--------|:-------:|:--------:|:--------:|
| Supervised | 0.9670 | 0.8681 | +0.0989 ❌ |
| **SimCLR** | 0.8736 | **0.8846** | **−0.0110** ✅ |
| SupCon | 0.9341 | 0.8846 | +0.0495 |
| BYOL | 0.9341 | 0.9176 | +0.0165 |
| ViT-Base | 0.9615 | 0.9286 | +0.0330 |

> **Most robust:** SimCLR — accuracy actually *improves* slightly on grayscale (−0.0110 drop)  
> **Least robust:** Supervised — largest drop of 9.9 percentage points

### Cosine Confidence: RGB vs Grayscale

| Method | RGB Cos | Gray Cos | Cos Drop |
|--------|:-------:|:--------:|:--------:|
| Supervised | 0.8169 | 0.7490 | +0.0679 |
| **SimCLR** | 0.7329 | 0.7351 | **−0.0022** ✅ |
| SupCon | 0.8051 | 0.7724 | +0.0327 |
| **BYOL** | 0.7070 | 0.7128 | **−0.0058** ✅ |
| ViT-Base | 0.5710 | 0.4861 | +0.0849 ❌ |

> SimCLR and BYOL show *negative* cosine drop — their feature space is inherently color-invariant.

### Per-Method Classification Reports (Grayscale Test)

<details>
<summary>Supervised — Grayscale</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.95      0.80      0.87        99
 NOCORROSION       0.80      0.95      0.87        83
    accuracy                           0.87       182
```
</details>

<details>
<summary>SimCLR — Grayscale</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.90      0.89      0.89        99
 NOCORROSION       0.87      0.88      0.87        83
    accuracy                           0.88       182
```
</details>

<details>
<summary>SupCon — Grayscale</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.91      0.87      0.89        99
 NOCORROSION       0.85      0.90      0.88        83
    accuracy                           0.88       182
```
</details>

<details>
<summary>BYOL — Grayscale</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.91      0.94      0.93        99
 NOCORROSION       0.93      0.89      0.91        83
    accuracy                           0.92       182
```
</details>

<details>
<summary>ViT-Base — Grayscale</summary>

```
              precision    recall  f1-score   support
   CORROSION       0.99      0.88      0.93        99
 NOCORROSION       0.87      0.99      0.93        83
    accuracy                           0.93       182
```
</details>

---

## 4. Overall Summary

| Method | ID Acc | Gray Acc | Feature AUROC | OOD Reject Rate | Notes |
|--------|:------:|:--------:|:-------------:|:---------------:|-------|
| Supervised | 96% | 87% | 0.830 | 64% | Best accuracy, least color-robust |
| SimCLR | 87% | **88%** | **0.839** | **68%** | Most color-robust; best OOD detector |
| SupCon | 93% | 88% | 0.819 | 59% | Solid balance |
| BYOL | 93% | **92%** | 0.757 | 54% | Strong grayscale; weakest OOD AUROC |
| ViT-Base | **96%** | **93%** | 0.820 | 49% | Best grayscale accuracy; best softmax AUROC |

### Key Takeaways

- **ViT-Base** achieves the highest grayscale accuracy (93%) and best softmax-based OOD detection (AUROC 0.8413), making it the strongest overall model.
- **SimCLR** is the most color-invariant method — grayscale shift causes essentially zero degradation — and achieves the best feature-space OOD AUROC (0.8391).
- **BYOL** is the second most robust to grayscale shift (only +1.6% drop) and maintains strong class separation.
- **Supervised ResNet-18** achieves top ID accuracy but is the most brittle under the grayscale domain shift (−9.9%).
- Self-supervised methods (SimCLR, BYOL) learn representations less tied to color, making them inherently more robust to color-based domain shifts.

---

## 5. Visualisations

| Figure | Description |
|--------|-------------|
| `grayscale_accuracy_drop.png` | Bar chart — RGB vs grayscale accuracy per method |
| `grayscale_cos_distributions.png` | Cosine score distributions (RGB vs gray) per method |
| `grayscale_pca_overlap.png` | PCA feature-space overlap of RGB and grayscale embeddings |
| `pca_id_separation.png` | ID feature-space class separation per method |
| `cos_threshold_plot.png` | Cosine threshold vs ID coverage / OOD reject rate |

---

## 6. Setup

```
dataset_ood/
  train/   CORROSION/ NOCORROSION/   (2 522 images)
  val/     CORROSION/ NOCORROSION/   (653 images)
  test/    CORROSION/ NOCORROSION/   (182 images)
  test_grayscale/  ...               (183 images, auto-generated)
```

Checkpoints: `sup_best.pt` · `simclr_best.pt` · `supcon_best.pt` · `byol_best.pt` · `vit_best.pt`

Notebooks:
- `comparison_all_methods_v2.ipynb` — training, linear probe, OOD detection
- `domain_shift_grayscale_v2.ipynb` — grayscale robustness evaluation

