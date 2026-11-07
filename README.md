# Adult Income Classification – ML Assignment 2

**BITS Pilani – M.Tech (DSE) – Machine Learning – Assignment 2**

## 1. Project Overview

This project builds an end-to-end machine learning workflow on the UCI Adult (Census Income) dataset: data cleaning and preprocessing, training five classification models, evaluating them with six metrics, and serving the results through an interactive Streamlit web application deployed on Streamlit Community Cloud.

## 2. Problem Statement

Predict whether a person's annual income exceeds **USD 50,000** based on demographic and employment attributes collected in the 1994 US Census. This is a **binary classification** problem:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | `<=50K` | Income less than or equal to $50K |
| 1 | `>50K` | Income greater than $50K |

## 3. Dataset Description

| Property | Value |
|----------|-------|
| Name | Adult (Census Income) |
| Source | UCI Machine Learning Repository |
| Instances (after removing 24 duplicates) | **32,537** (requirement: ≥ 500) |
| Input features | **14** (requirement: ≥ 12) – 6 numeric, 8 categorical |
| Target | `income` (`<=50K` / `>50K`) |
| Class distribution | 75.9 % `<=50K`, 24.1 % `>50K` (imbalanced) |
| Missing values | `workclass` (1,836), `occupation` (1,843), `native_country` (582) – marked `?` |

**Features**

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age in years |
| workclass | Categorical | Employer type (Private, Self-emp, Government, …) |
| fnlwgt | Numeric | Census sampling weight |
| education | Categorical | Highest education level |
| education_num | Numeric | Education level as a number |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Occupation type |
| relationship | Categorical | Relationship within household |
| race | Categorical | Race |
| sex | Categorical | Sex |
| capital_gain | Numeric | Capital gains |
| capital_loss | Numeric | Capital losses |
| hours_per_week | Numeric | Working hours per week |
| native_country | Categorical | Country of origin |

## 4. Dataset Source

- UCI Machine Learning Repository – Adult: https://archive.ics.uci.edu/dataset/2/adult
- Raw file used: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

The training script downloads the file automatically on first run and stores the cleaned copy in `data/dataset.csv`.

## 5. GitHub Repository Link

`https://github.com/guptayush786/ML_Assignment_2`

**Live Streamlit App:** `https://mlassignment2-e3ytasps8ctwaujqbcnxrm.streamlit.app/`

## 6. Data Preprocessing Steps

1. **Cleaning** – stripped leading spaces from values, converted `?` to missing values, removed 24 duplicate rows.
2. **Target encoding** – `>50K` → 1, `<=50K` → 0.
3. **Train–test split** – 80 % / 20 % stratified split with `random_state=42` (26,029 train / 6,508 test rows). The same split is used for all five models; the test split is saved as `test_data.csv`.
4. **Missing value handling** – numeric: median imputation; categorical: most-frequent imputation.
5. **Categorical encoding** – One-Hot Encoding (`handle_unknown="ignore"` so unseen categories in uploaded data do not break prediction).
6. **Feature scaling** – `StandardScaler` on numeric features for Logistic Regression, kNN, Naive Bayes and SVM. Not applied for the Decision Tree because trees are scale-invariant.

All preprocessing is wrapped in a scikit-learn `Pipeline` together with the classifier, so each saved model accepts raw CSV columns directly.

## 7. Models Implemented

| # | Model | Key settings |
|---|-------|--------------|
| 1 | Logistic Regression | `max_iter=2000` |
| 2 | Decision Tree Classifier | `max_depth=10`, `min_samples_leaf=20` |
| 3 | K-Nearest Neighbors | `n_neighbors=15` |
| 4 | Naive Bayes | Gaussian Naive Bayes |
| 5 | Support Vector Machine | RBF kernel, `C=1.0` |

## 8. Evaluation Metrics

Computed on the held-out test set (positive class = `>50K`):

1. **Accuracy** – fraction of correct predictions
2. **AUC** – area under the ROC curve (uses predicted probabilities; SVM uses decision-function scores)
3. **Precision** – of the people predicted `>50K`, how many truly earn `>50K`
4. **Recall** – of the people truly earning `>50K`, how many were found
5. **F1 Score** – harmonic mean of precision and recall
6. **MCC** – Matthews Correlation Coefficient, a balanced measure robust to class imbalance

## 9. Model Comparison Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8569 | 0.9099 | 0.7437 | 0.6199 | 0.6762 | 0.5893 |
| Decision Tree | **0.8666** | **0.9106** | **0.7961** | 0.6001 | **0.6844** | **0.6116** |
| kNN | 0.8459 | 0.8903 | 0.7016 | 0.6269 | 0.6622 | 0.5643 |
| Naive Bayes | 0.5361 | 0.7492 | 0.3369 | **0.9560** | 0.4983 | 0.3300 |
| Support Vector Machine | 0.8586 | 0.9066 | 0.7630 | 0.5995 | 0.6714 | 0.5899 |

(Saved in `metrics/model_metrics.csv`.)

## 10. Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong and stable baseline (Accuracy 0.857, AUC 0.910 – second best). One-hot encoding lets a linear boundary separate the classes well. Train accuracy (0.851) ≈ test accuracy, so no overfitting. Recall of 0.62 shows the minority `>50K` class is harder to detect. |
| Decision Tree | Best model overall (Accuracy 0.867, AUC 0.911, Precision 0.796, MCC 0.612). Captures non-linear interactions such as marital status × capital gain. Limiting depth to 10 and leaf size to 20 prevented overfitting (train 0.863 vs test 0.867). |
| kNN | Lowest of the four strong models (Accuracy 0.846, AUC 0.890). Distances become less informative in the high-dimensional one-hot space, and it shows a small overfitting gap (train 0.858). Prediction is also slowest because every test point is compared with all training points. |
| Naive Bayes | Weakest model (Accuracy 0.536, MCC 0.330). The independence and Gaussian assumptions are violated by binary one-hot features and highly skewed capital gain/loss, so it predicts `>50K` far too often – very high recall (0.956) but very low precision (0.337). |
| Support Vector Machine | Comparable to Logistic Regression (Accuracy 0.859, AUC 0.907, Precision 0.763) with good generalisation (train 0.865). The RBF kernel adds little over a linear boundary here, while training is the most computationally expensive. Lowest recall (0.600) among the strong models. |
| **Overall, Winner for your dataset?** | **Decision Tree** – it ranks first on 5 of the 6 metrics (Accuracy, AUC, Precision, F1, MCC). |

## 11. Best Model Analysis

- **Best Accuracy model:** Decision Tree (0.8666)
- **Best AUC model:** Decision Tree (0.9106)
- **Overall best model (best average rank across all six metrics):** Decision Tree

The depth-limited Decision Tree models the threshold-type rules in this data (e.g. high capital gain, married-civilian-spouse relationship, higher education) directly, without needing feature scaling, and its regularisation keeps it from overfitting. Logistic Regression and SVM are close behind (within ~1 % accuracy), which shows the problem is largely linearly separable after one-hot encoding. All models have lower recall than precision on the `>50K` class because of the 76/24 class imbalance; MCC is therefore the most reliable single comparison metric, and the Decision Tree also leads on MCC.

## 12. Project Structure

```
project-folder/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test_data.csv              # Held-out test split (6,508 rows) for the app
├── assignment_report_content.md
├── data/
│   └── dataset.csv            # Cleaned full dataset
├── model/
│   ├── train_models.py        # Preprocessing, training, evaluation
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
└── metrics/
    └── model_metrics.csv      # Comparison table of all 6 metrics
```

## 13. Setup Instructions

Requires Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

To retrain all models and regenerate `test_data.csv` and `metrics/model_metrics.csv` (takes about 1–3 minutes):

```bash
python model/train_models.py
```

## 14. How To Run The Application

```bash
streamlit run app.py
```

Then in the browser:

1. **Upload** `test_data.csv` from the sidebar (a download button for it is also provided in the app).
2. **Choose a model** from the dropdown.
3. **Selected model** tab – six evaluation metrics, confusion matrix and classification report.
4. **Model comparison** tab – all five models evaluated on the uploaded data, plus the saved benchmark table.
5. **Predictions** tab – row-level predictions with a CSV download.

## 15. Streamlit Deployment Steps

1. Push this project to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **Create app** / **New app**.
4. Select the repository, branch `main`, and main file `app.py`.
5. Under *Advanced settings*, choose Python **3.12**.
6. Click **Deploy** and wait for the dependencies to install.

## Next Steps

1. **Create a GitHub repository** – Log in to https://github.com → click **New** → give it a name (e.g. `ml-assignment-2`) → choose **Public** → click **Create repository**. Do not add a README there (we already have one).
2. **Push the code** – Open a terminal in the project folder and run:
   ```bash
   git init
   git add .
   git commit -m "ML Assignment 2: Adult income classification"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   Then paste the repository link into Section 5 of this README and push again.
3. **Deploy on Streamlit Cloud** – Follow Section 15. After deployment you get a link like `https://<something>.streamlit.app`. Open it, upload `test_data.csv`, and check that everything works. Add this link to Section 5.
4. **Run once in BITS Virtual Lab** – Log in to the BITS Virtual Lab, open a terminal, then:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   pip install -r requirements.txt
   python model/train_models.py
   streamlit run app.py
   ```
5. **Capture the screenshot** – In the Virtual Lab, take ONE screenshot that shows the terminal with the training output (metrics table) and/or the running Streamlit app in the browser, with the Virtual Lab environment visible.
6. **Submit** – Create ONE PDF containing, in this order: (1) GitHub repository link, (2) live Streamlit app link, (3) the BITS Virtual Lab screenshot, (4) the content of this README. Upload and press **Submit** (not draft) before the deadline.
