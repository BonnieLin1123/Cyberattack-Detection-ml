# Cyberattack Detection ml - Using Transfer Learning: Fine-Tuning ResNet-18 on Network Traffic Data

**Authors:** TzuChen (Bonnie) Lin, HsiangEn (Shawna) Liu
**Course:** DS340 Machine Learning — Professor Kevin Gold
**Date:** April 27, 2026

[Link to Colab Notebook](https://colab.research.google.com/drive/1rKbMLPdciBNV2YSoCrllWk5QQ0ttJRJM)

## Abstract

This project applies transfer learning from computer vision to tabular network security data for intrusion detection. We fine-tune **ResNet-18**, a convolutional neural network pretrained on ImageNet, to classify network connections from the **NSL-KDD** dataset into attack categories. Our core innovation is a **feature-to-image conversion pipeline** that reshapes each connection's 41 network traffic features into a 32×32 pixel grid and feeds it into ResNet-18.

Through a systematic five-step experimental progression addressing class imbalance via class weighting, SMOTE oversampling, and class redefinition, our best model achieves a **macro F1 score of 0.7376** on the standard KDDTest+ benchmark — a **25% improvement** over the Random Forest baseline (0.5915) — with reasonable generalization to the harder KDDTest-21 set (**0.6692**).

## 1. Introduction

Traditional rule-based intrusion detection systems struggle to adapt to new and evolving attack patterns. Machine learning offers an alternative by learning attack patterns directly from data.

The **NSL-KDD** dataset is a well-established network intrusion detection benchmark, derived from a simulation of a real US Air Force network with injected attacks. Each row represents a network connection described by 41 features (duration, protocol type, failed logins, bytes transferred, etc.), classified into one of five categories:

- **Normal** — legitimate traffic
- **DoS** — Denial of Service (flood/crash attacks)
- **Probe** — network scanning
- **R2L** — Remote to Local (unauthorized remote access)
- **U2R** — User to Root (privilege escalation)

This project investigates whether a CNN pretrained on everyday photographs can be repurposed to detect cyberattacks in network traffic by converting tabular feature vectors into image-like grids.

A key secondary challenge is **severe class imbalance** — R2L and U2R together make up less than 1% of training examples (U2R: 52 samples, R2L: 995 samples, vs. 67,343 Normal samples). Our experiments systematically evaluate strategies to address this.

## 2. Background

- **ResNet-18**: An 18-layer CNN using residual (skip) connections to solve the vanishing gradient problem, pretrained on 1.2M ImageNet images. Its learned edge/texture/shape detectors transfer surprisingly well to our feature-grid images.
- **SMOTE** (Synthetic Minority Over-sampling Technique): Generates synthetic minority-class examples by interpolating between nearest neighbors.
- **Macro F1 Score**: Averages F1 across all classes equally, weighting rare classes the same as common ones — more informative than accuracy on imbalanced data.

## 3. Methodology

### 3.1 Dataset

| Split | Samples |
|---|---|
| KDDTrain+ | 125,973 |
| KDDTest+ | 18,794 |
| KDDTest-21 (hard subset) | 8,110 |

| Class | Count | Percentage | Description |
|---|---|---|---|
| Normal | 67,343 | 53.5% | Legitimate traffic |
| DoS | 45,927 | 36.5% | Flood/crash attacks |
| Probe | 11,656 | 9.3% | Network scanning |
| R2L | 995 | 0.8% | Unauthorized remote access |
| U2R | 52 | 0.04% | Privilege escalation |

U2R is ~1,300x rarer than Normal traffic — a naive classifier can hit 53% accuracy while never detecting a single attack.

### 3.2 Preprocessing

1. **Label mapping** — 23 raw attack labels grouped into 5 broad classes.
2. **Categorical encoding** — `protocol_type`, `service`, `flag` encoded via `LabelEncoder` (fit on train only).
3. **Normalization** — All 41 features scaled to [0, 1] via `MinMaxScaler` (fit on train only).
4. **Feature-to-image conversion**:
   - Pad 41 features → 49 (append 8 zeros)
   - Reshape into a 7×7 grid
   - Resize to 32×32 via bilinear interpolation
   - Repeat channel 3× → RGB tensor `(3, 32, 32)`

Different attack classes produce visibly distinct spatial patterns (e.g., DoS shows high byte-count features; U2R shows elevated privilege-related features).

### 3.3 Models

- **Random Forest** (100 trees, bootstrap sampling) and **Linear SVM** — classical baselines on raw 41-feature vectors.
- **ResNet-18** — final FC layer replaced to output 5 classes (or 4, in Step 5), all other weights initialized from ImageNet pretraining.

### 3.4 Two-Stage Fine-Tuning

| Stage | Epochs | LR | Description |
|---|---|---|---|
| 1 | 5 | 0.001 | Pretrained layers frozen; only new head trained |
| 2 | 10 | 0.0001 | Full network fine-tuned end-to-end |

### 3.5 Class Imbalance Strategies

- **Class weights**: Inverse-frequency penalty (U2R weight ≈ 484.5)
- **SMOTE**: Synthetic oversampling of minority classes (train only)
- **Gentle class weights** (SMOTE + weights): Small nudge on top of SMOTE-balanced data — `[1.0, 1.0, 1.0, 2.0, 3.0]`

**Five-step experimental progression:**

1. Baselines (RF + SVM) — no image conversion, no imbalance handling
2. CNN + Class Weights
3. CNN + SMOTE
4. CNN + SMOTE + Gentle Weights
5. CNN, 4-class (R2L + U2R merged into "Others") + SMOTE + Gentle Weights, evaluated on both test sets

## 4. Results

### 4.1 Baselines (Step 1)

| Class | RF Precision | RF Recall | RF F1 | SVM F1 |
|---|---|---|---|---|
| Normal | 0.82 | 0.97 | 0.89 | 0.87 |
| DoS | 0.99 | 0.99 | 0.99 | 0.95 |
| Probe | 0.85 | 1.00 | 0.92 | 0.87 |
| R2L | 0.99 | 0.06 | 0.11 | 0.00 |
| U2R | 0.50 | 0.03 | 0.05 | 0.27 |
| **Macro F1** | | | **0.59** | **0.59** |

Both baselines almost completely fail on rare classes (SVM never predicts R2L at all).

### 4.2 CNN + Class Weights (Step 2)
Macro F1: **0.64**. R2L F1 jumped from 0.11 → 0.48, but U2R precision collapsed to 0.13 due to over-prediction from the extreme 484x weight.

### 4.3 CNN + SMOTE (Step 3)
Macro F1: **0.66**. Training set expanded from 125,973 → 336,715 samples (U2R: 1,294x synthetic multiplication). U2R F1 rose to 0.29, but R2L F1 declined to 0.41.

### 4.4 CNN + SMOTE + Gentle Weights (Step 4)
Macro F1: **0.66** (no clear improvement over Step 3). Best U2R F1 in the 5-class setting (0.35), but R2L F1 dropped to 0.34.

### 4.5 CNN 4-class + SMOTE + Gentle Weights (Step 5) — Best Model

Merging R2L + U2R into "Others" reduced the SMOTE multiplication ratio from 1,294x to 63x.

| Class | KDDTest+ F1 | KDDTest-21 F1 |
|---|---|---|
| Normal | 0.89 | 0.62 |
| DoS | 0.98 | 0.96 |
| Probe | 0.77 | 0.78 |
| Others | 0.32 | 0.32 |
| **Macro F1** | **0.74** | **0.67** |

Others precision reached 0.94 (few false alarms), but recall stayed at 0.19, reflecting the difficulty of detecting ~1,047 minority examples among 67,343 Normal samples.

### 4.6 Summary of All Models

| Model | R2L/Others F1 | U2R F1 | Macro F1 |
|---|---|---|---|
| Random Forest | 0.11 | 0.05 | 0.59 |
| SVM | 0.00 | 0.27 | 0.59 |
| CNN + Class Weights | 0.48 | 0.13 | 0.64 |
| CNN + SMOTE | 0.41 | 0.29 | 0.66 |
| CNN + SMOTE + Gentle Weights | 0.34 | 0.35 | 0.66 |
| **CNN 4-class (KDDTest+)** | 0.32 | – | **0.74** |
| CNN 4-class (KDDTest-21) | 0.32 | – | 0.67 |

## 5. Conclusions

A pretrained CNN can be effectively repurposed for network intrusion detection via feature-to-image conversion. The best model — ResNet-18 fine-tuned on 4-class NSL-KDD with SMOTE + gentle class weights — achieved a macro F1 of **0.7376** on KDDTest+, a 25% improvement over the Random Forest baseline.

Key takeaways:
- No single imbalance-handling strategy solved rare-class detection outright — each involved a distinct tradeoff (e.g., class weights improved R2L but hurt U2R precision; SMOTE helped U2R but hurt R2L).
- **Redefining class granularity** (merging R2L + U2R into "Others") produced a bigger improvement than any individual reweighting/oversampling technique, suggesting class structure choices can matter as much as model architecture.
- Rare-attack detection remains an open problem — Others recall was only 0.19 even in the best model.

**Future work:** spatially-aware feature ordering in the grid conversion, focal loss as an alternative to fixed class weights, ensemble methods combining ResNet-18 and Random Forest, and evaluation on additional intrusion detection datasets for cross-dataset generalization.

## References

- Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed analysis of the KDD CUP 99 data set.* IEEE Symposium on Computational Intelligence for Security and Defense Applications.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique.* Journal of Artificial Intelligence Research, 16, 321–357.
- Russakovsky, O., et al. (2015). *ImageNet large scale visual recognition challenge.* International Journal of Computer Vision, 115(3), 211–252.
- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). *Focal loss for dense object detection.* ICCV.
