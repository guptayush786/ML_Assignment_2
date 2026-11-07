# Machine Learning Assignment 2 – Report Content

**Programme:** M.Tech (Data Science & Engineering), BITS Pilani – WILP
**Course:** Machine Learning
**Topic:** Income Classification on the UCI Adult (Census Income) Dataset

**GitHub Repository:** `<paste GitHub repository URL>`
**Live Streamlit App:** `<paste Streamlit app URL>`

---

## 1. Problem Statement

The objective of this work is to predict whether an individual's annual income exceeds USD 50,000 using demographic and employment attributes from the 1994 United States Census. The task is formulated as a supervised binary classification problem, where the positive class (`>50K`) represents high-income individuals and the negative class (`<=50K`) represents all others. Five classification algorithms are implemented on the same dataset and compared using six standard evaluation metrics, and the trained models are made available through an interactive Streamlit web application.

## 2. Dataset Description

The Adult (Census Income) dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/2/adult).

| Property | Value |
|----------|-------|
| Instances | 32,537 (after removal of 24 duplicate records) |
| Input features | 14 (6 numeric, 8 categorical) |
| Target variable | `income` – `<=50K` (75.9 %) / `>50K` (24.1 %) |
| Missing values | `workclass` (1,836), `occupation` (1,843), `native_country` (582) |

The numeric features are *age, fnlwgt, education_num, capital_gain, capital_loss* and *hours_per_week*. The categorical features are *workclass, education, marital_status, occupation, relationship, race, sex* and *native_country*. The dataset satisfies the assignment requirements of at least 12 features and 500 instances. The class distribution is moderately imbalanced, which motivates the use of class-sensitive metrics such as F1 score and MCC alongside accuracy.

## 3. Data Preprocessing Summary

1. **Data cleaning:** leading whitespace was stripped from all values, the `?` placeholder was converted to missing values, and 24 duplicate rows were removed.
2. **Target encoding:** `>50K` was mapped to 1 and `<=50K` to 0.
3. **Train–test split:** an 80:20 stratified split with a fixed random seed (42) produced 26,029 training and 6,508 test instances. The identical split was used for all models, and the test partition was exported as `test_data.csv`.
4. **Missing value imputation:** median imputation for numeric features and most-frequent imputation for categorical features, fitted on the training data only.
5. **Categorical encoding:** one-hot encoding, configured to ignore categories unseen during training.
6. **Feature scaling:** standardisation (zero mean, unit variance) of numeric features for the distance- and gradient-based models (Logistic Regression, kNN, SVM) and for Gaussian Naive Bayes. Scaling was omitted for the Decision Tree, which is invariant to monotonic feature transformations.

Preprocessing and classification were combined in a single scikit-learn `Pipeline` per model, preventing data leakage and ensuring identical transformations at training and inference time.

## 4. Models Used

| Model | Configuration |
|-------|---------------|
| Logistic Regression | L2-regularised, `max_iter=2000` |
| Decision Tree Classifier | `max_depth=10`, `min_samples_leaf=20` |
| K-Nearest Neighbour Classifier | `k=15`, Euclidean distance |
| Naive Bayes Classifier | Gaussian Naive Bayes |
| Support Vector Machine | RBF kernel, `C=1.0` |

## 5. Metrics Comparison Table

All metrics were computed on the held-out test set (6,508 instances), with `>50K` as the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8569 | 0.9099 | 0.7437 | 0.6199 | 0.6762 | 0.5893 |
| Decision Tree | **0.8666** | **0.9106** | **0.7961** | 0.6001 | **0.6844** | **0.6116** |
| kNN | 0.8459 | 0.8903 | 0.7016 | 0.6269 | 0.6622 | 0.5643 |
| Naive Bayes | 0.5361 | 0.7492 | 0.3369 | **0.9560** | 0.4983 | 0.3300 |
| Support Vector Machine | 0.8586 | 0.9066 | 0.7630 | 0.5995 | 0.6714 | 0.5899 |

## 6. Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides a strong, well-generalised baseline (Accuracy 0.857, AUC 0.910). After one-hot encoding, the classes are largely linearly separable. Training and test accuracy are nearly identical (0.851 vs 0.857), indicating no overfitting; however, recall on the minority class is moderate (0.620). |
| Decision Tree | Achieves the best Accuracy (0.867), AUC (0.911), Precision (0.796), F1 (0.684) and MCC (0.612). It captures non-linear feature interactions and threshold effects (e.g. capital gain, marital status). Depth and leaf-size constraints effectively controlled variance, with no observable overfitting (train 0.863 vs test 0.867). |
| kNN | Delivers the lowest performance among the four competitive models (Accuracy 0.846, AUC 0.890). Euclidean distance is less discriminative in the high-dimensional sparse one-hot feature space, a small generalisation gap is observed (train 0.858), and inference cost scales with the size of the training set. |
| Naive Bayes | Performs poorly (Accuracy 0.536, MCC 0.330). The conditional-independence and Gaussian assumptions are violated by binary one-hot features and heavily skewed numeric variables, causing systematic over-prediction of the positive class: very high recall (0.956) but low precision (0.337). |
| Support Vector Machine | Performs comparably to Logistic Regression (Accuracy 0.859, AUC 0.907, Precision 0.763) with good generalisation. The RBF kernel provides no material advantage over a linear decision boundary on this dataset, while incurring the highest training cost; its recall (0.600) is the lowest among the competitive models. |
| **Overall, Winner for your dataset?** | **Decision Tree** – it ranks first on five of the six metrics. |

## 7. Best Model Analysis

- **Best Accuracy:** Decision Tree (0.8666)
- **Best AUC:** Decision Tree (0.9106)
- **Overall best (mean rank across all six metrics):** Decision Tree

The depth-limited Decision Tree is the most effective classifier for this dataset. Its ability to represent axis-aligned threshold rules matches the structure of the census attributes, and regularisation through `max_depth` and `min_samples_leaf` prevents overfitting. Logistic Regression and SVM follow closely (within approximately one percentage point of accuracy), confirming that the problem is close to linearly separable in the encoded feature space. Because of the 76:24 class imbalance, all competitive models exhibit higher precision than recall for the `>50K` class; MCC, which accounts for all four confusion-matrix cells, therefore provides the most reliable single comparison, and the Decision Tree also achieves the highest MCC (0.612).

## 8. Conclusion

Five classification models were trained and evaluated on the UCI Adult dataset using a consistent preprocessing pipeline and a common stratified train–test split. Tree-based, linear and kernel methods all achieved accuracy above 85 % and AUC above 0.90, whereas Gaussian Naive Bayes was unsuitable due to violated distributional assumptions, and kNN was limited by the high-dimensional encoded feature space. The Decision Tree emerged as the best overall model, combining the highest accuracy, AUC, precision, F1 score and MCC with good generalisation. The trained pipelines are deployed in a Streamlit application that allows users to upload test data, select a model, and inspect evaluation metrics, the confusion matrix, the classification report and a cross-model comparison. Future improvements could include handling class imbalance (class weighting or resampling), hyperparameter tuning with cross-validation, and ensemble methods such as Random Forest or Gradient Boosting.
