# Deep Learning & ANN Assignment Suite
### Comprehensive Implementation Report: Bank Churn, Loan Approval & Medical Insurance Cost Analysis

---

## 📌 Executive Summary & Architecture Paradigm
This repository contains a full-stack engineering suite of three core Artificial Neural Network (ANN) assignments. The projects systematically navigate from structural **Binary Classification** setups to continuous mathematical **Regression Frameworks**. 

Across all implementation cycles, a rigorous data preprocessing protocol was maintained to isolate data pipelines, enforce standard scaling vectors without leaking mathematical dependencies into unseen test partitions, and dynamically handle character-level structural anomalies.

---

## 🛠️ Global Preprocessing & Safety Framework
Before layer injection, data undergoes a stringent sequence designed to completely eliminate execution runtime exceptions and data contamination traps:

1. **Structural Whitespace Eradication:** Trailing and leading hidden spaces inside categorical rows are thoroughly neutralized using vector string stripping pipelines.
2. **Strict Split-Before-Scale Sequence:** To maintain standard statistical layouts, datasets are partitioned into separate matrices prior to feature normalization.
3. **Zero Data Leakage Boundary:** `StandardScaler` transformations learn internal configuration weights (`.fit_transform()`) strictly using the isolated training boundaries (`X_train`), while unseen test vectors (`X_test`) are strictly translated using mapping configurations (`.transform()`). This prevents structural features magnitude leakage into evaluation checkpoints.

---

## 🗂️ Detailed Project Portfolios

### 📈 Phase 1: Bank Customer Churn Prediction (Classification)
* **Objective:** Build a predictive classification network to identify high-risk banking customers before attrition occurs.
* **Problem Type:** Binary Classification (Target: `Exited` $\in \{0, 1\}$).
* **Architectural Profile:**
  * **Input Layout:** Dynamic reflection based on input column feature matrices (`input_dim=X_train.shape[1]`).
  * **Hidden Topography:** 16 Hidden Dense Units $\rightarrow$ 8 Hidden Dense Units utilizing the Rectified Linear Activation (`relu`) function to parse deep non-linear interaction terms.
  * **Output Layer Configuration:** 1 Dense Unit governed by a **Sigmoid Activation Function**, squashing complex matrix probabilities tightly into definitive absolute confidence constraints $[0, 1]$.
  * **Loss Optimization Metrics:** Compiled using the **Binary Cross-Entropy (`loss='binary_crossentropy'`)** objective function alongside the **Adam Optimizer**, tracking structural categorical metrics (`metrics=['accuracy']`).

---

### 🏛️ Phase 2: Credit Loan Approval Optimization (Classification)
* **Objective:** Design an automated micro-lending verification platform capable of evaluating structural applicant features to approve or reject internal loan disbursements.
* **Problem Type:** Binary Classification (Target: `loan_status` mapped to `Approved: 1`, `Rejected: 0`).
* **Technical Challenges & Resolutions:**
  * Encountered standard runtime failures (`KeyError: 'education'` and `IndexError`) due to hidden spaces inside tabular row features. Resolved globally by explicitly stripping data columns using `.str.strip()`.
  * **Categorical Mapping Strategies:** Features transformed into machine-readable numeric formats via explicit integer injection:
    * `education` $\rightarrow$ `{'Graduate': 1, 'Not Graduate': 0}`
    * `self_employed` $\rightarrow$ `{'Yes': 1, 'No': 0}`
    * `loan_status` $\rightarrow$ `{'Approved': 1, 'Rejected': 0}`
  * Removed identity features (`loan_id`) to ensure the neural weights prioritize pure behavioral analytics over arbitrary sequencing metadata.
* **Performance Analysis:**
  * Achieved a stable **92.19% Test Accuracy** with an optimized validation curve that showed no tracking deviations or overfitting symptoms.
  * **Confusion Matrix Profiling:**
    * **True Negatives (TN):** 427 loans correctly rejected.
    * **True Positives (TP):** 754 loans correctly approved.
    * **False Positives (FP):** 44 exceptions where high-risk parameters were structurally bypassed.
    * **False Negatives (FN):** 56 standard loans overly penalized.

---

### 🏥 Phase 3: Medical Insurance Price Optimization (Regression)
* **Objective:** Model a financial projection environment to calculate individualized medical premium liabilities based on demographic profiles.
* **Problem Type:** Continuous Parametric Regression (Target: `charges`).
* **Architectural Pivot Strategy (The Classification-to-Regression Shift):**
  * **Output Layer Transformation:** Configured 1 Dense Neuron with **No Activation Function (Linear)** to allow unrestrained, continuous dollar predictions, dropping the categorical Sigmoid function.
  * **Loss Function Realignment:** Replaced binary cross-entropy with **Mean Squared Error (MSE)** (`loss='mse'`) to geometrically penalize large financial pricing errors during backward gradient distributions.
  * **Evaluation Metric Swap:** Implemented **Mean Absolute Error (MAE)** (`metrics=['mae']`) to calculate absolute dollar errors rather than accuracy percentages.
* **Hyperparameter Tuning & Optimization Curves:**
  * Moving from 20 to 500 training epochs with higher neuron distributions ($32 \rightarrow 16 \rightarrow 1$) decreased the MAE from **$13,106** down to an optimized baseline of **$3,653** on training matrices and **$3,770** on test profiles.
  * **R2 Score Integration:** Achieved an $R^2$ evaluation score of **0.8035**, indicating that the network accounts for **80.35%** of the underlying financial variation governing healthcare premium costs.

---

## 📈 Theoretical Synthesis: The Learning Curve & Early Stopping Paradigm
The implementation explicitly mapping training runs across 500 epochs validated key deep learning behaviors:

* **The Elbow Point Rule:** Visualization profiles confirmed that the loss curve tracks a steep descent from epochs 0 to 100 before reaching a distinct inflection point ("The Elbow") between epochs 100 and 150. Beyond this baseline, the learning velocity drops significantly, and performance scores flatten out.
* **The Overfitting Threshold:** Pushing epochs up to 5000 without early stopping parameters triggers memory memorization patterns. While training loss mathematically drops near zero, validation distance metrics (`val_loss`) diverge upward. This is avoided here because training error ($3,653$) and validation error ($3,770$) track closely together, ensuring robust real-world generalization.

---
*Prepared as part of the academic deep learning submission matrix for the  Artificial Intelligence curriculum.*
