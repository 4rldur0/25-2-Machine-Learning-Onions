# Website Fingerprinting Attack on Tor Network

## 1\. Team Introduction

  * **Team Name:** Onions
  * **Members:**
      * 성유정 (2171020): Team Leader
      * 박나연 (2270097)
      * 김나경 (2271010)
      * Mi Luoni (IES25316)
      * Glenn Lim (IES25588)

## 2\. Project Overview

This project implements a **Website Fingerprinting (WF) attack** on the [Tor](https://en.wikipedia.org/wiki/Tor_(network)) anonymous network. We aim to classify encrypted traffic traces to identify which website a user is visiting. The solution covers the full machine learning pipeline: feature extraction, model training (including hyperparameter tuning), and evaluation under both Closed-world and Open-world scenarios.

  * **Key Strategy:** Utilizing a **Stacking Ensemble** method combining multiple base learners (RF, XGB, LGBM, CAT, etc.) to maximize classification performance.

## 3\. Repository Structure

```bash
.
├── 01_Preprocessing_33 features.ipynb  # [Step 1] Feature extraction & Data split
├── 02_Model_Training_Binary.ipynb      # [Step 2-1] Training for Binary classifier
├── 02_Model_Training_Multi.ipynb       # [Step 2-2] Training for Multi-class Classifier
├── 03_Final_Evaluation_Closed.ipynb    # [Step 3-1] Evaluation metrics for Closed-world
├── 03_Final_Evaluation_Open.ipynb      # [Step 3-2] Evaluation metrics for Open-world
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── data
│   ├── raw                             # [Input] Place original data files here
│   └── preprocessed                    # [Output] Generated .pkl files (Train/Val/Test)
├── experiments                         # Draft notebooks for model selection & tuning
│   ├── binary                          # Experiments for binary classifiers (RF, XGB, LR...)
│   └── multi                           # Experiments for multi-class classifiers (CAT, KNN, NN...)
├── models                              # Saved trained models (.pkl)
│   ├── binary                          # Binary models
│   └── multi                           # Multi-class models
└── src                                 # Source code modules (Helper functions)
    ├── evaluation_binary.py            # Metrics & plotting for binary tasks
    ├── evaluation_multi.py             # Metrics & plotting for multi-class tasks
    ├── evaluation_final.py             # Evaluation Piplelines for both scenarios (Closed/Open)
    └── train_tuning.py                 # Training & GridSearch logic
```

## 4\. How to Run

### Step 0: Environment Setup

A. Install Dependencies
Install the required Python packages.

```bash
pip install -r requirements.txt
```

B. Configure Path Settings
Before running any notebook, you must configure the root path according to your environment. Open each notebook and modify the Settings cell at the beginning:
```python
# ⚠️ Uncomment if running on Goggle Colab
# from google.colab import drive 
# drive.mount('/content/drive')

ROOT = "/content/drive/25-2-Machine-Learning-Onions/"  # ⚠️ Change this to your project path
import sys
sys.path.append(ROOT)
```

### Step 1: Data Preparation & Preprocessing

1.  Place the raw dataset pickle files into the `data/raw/` directory.
      * Monitored dataset file is assumed to be `mon_standard.pkl`.
      * Unmonitored dataset file is assumed to be `unmon_standard10.pkl`.
      * Should the files be named differently, change the target filepath in the notebook first.
3.  Run **`01_Preprocessing_33 features.ipynb`**.
      * This notebook extracts **33 statistical features** (packet size, timing, bursts, etc.) from the raw data.
      * It splits the data and saves processed files into `data/preprocessed/`.
      * Output:
        * `close_train_33.pkl`, `close_val_33.pkl`, `close_test_33.pkl` (for Multi-class Model)
        * `open_train_33.pkl`, `open_val_33.pkl`, `open_test_33.pkl` (for Binary Model)

### Step 2: Model Training

Run the training notebooks to build and save the models.

  * **For Open-world (Binary Classification):**

      * Run **`02_Model_Training_Binary.ipynb`**.
      * This trains binary classifiers (including Stacking Ensemble) to distinguish between monitored and non-monitored traffic.
           * This notebook trains the models using data from `open_train_33.pkl`, and conducts performance tests using data from `open_val_33.pkl`.
      * Trained models are saved to `models/binary/`.

  * **For Both worlds (Multi-class Classification):**

      * Run **`02_Model_Training_Multi.ipynb`**.
      * This trains the multi-class classifier (including Stacking Ensemble) to identify specific websites.
            * This notebook trains the models using data from `close_train_33.pkl`, and conducts performance tests using data from `close_val_33.pkl`.
      * The final model is saved to `models/multi/`.

### Step 3: Evaluation

Run the evaluation notebooks to generate performance reports and visualization.

  * **Closed-world Scenario:**

      * Run **`03_Final_Evaluation_Closed.ipynb`**.
      * **Metrics:** Confusion Matrix, Per-class Accuracy, F1
           * This notebook evaluates the models against data from `close_test_33.pkl`.

  * **Open-world Scenario:**

      * Run **`03_Final_Evaluation_Open.ipynb`**.
      * **Metrics:** Confusion Matrix, Per-class Accuracy, F1 for each stage
            * This notebook evaluates the models against data from `open_test_33.pkl`.
      * This notebook implements the **two-stage approach**.
           * Stage 1: Filtering non-monitored traffic using the Binary Model - labelled with -1.
           * Stage 2: Classifying the remaining traffic using the Multi-class Model - labelled with one of 0 through 94.

## 5\. Methodology & Experiment Details

We explored various algorithms before finalizing our models. The detailed logs of these experiments can be found in the `experiments/` folder.

  * **Feature Set:** Max 33 features (Packet counts, Inter-arrival times, Bursts, etc.).
  * **Binary Models Explored:** Random Forest, XGBoost, Logistic Regression, LGBM, CatBoost, SVM.
  * **Multi-class Models Explored:** Random Forest, XGBoost, Logistic Regression, LGBM, CatBoost, KNN, Neural Networks, SVM.
  * **Final Selection:** We selected the best-performing models based on validation accuracy and FPR.
