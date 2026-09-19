# 🎯 Smart Recruitment Assistant

A machine learning project that predicts which candidates are likely to be open to a job change, so recruiters can prioritize who to contact first.

Built with **Python, Pandas, Scikit-learn, Matplotlib and Seaborn** as part of my AI training at the Information Technology Institute (ITI).

---

## 📌 Problem

Recruiters often spend time reaching out to candidates who are not looking to move. This project uses candidate data to estimate the **probability that a candidate will change jobs**, and ranks candidates accordingly.

## 📂 Dataset

**HR Analytics: Job Change of Data Scientists** (public dataset)

- 19,158 raw records, **19,109** after removing duplicates
- 13 candidate features: city development index, education, experience, company size/type, training hours, last job change, and more
- Target: `1` = looking for a job change, `0` = not looking
- Imbalanced: **~75% vs ~25%**

![Target distribution](images/05_recruitment_statistics.png)

## ⚙️ Workflow

1. **Data preparation**
   - Removed duplicates and the `enrollee_id` column
   - Filled missing categorical values with `Unknown` and the rest with the mode
   - Added missing-value indicators for `company_size` and `company_type`
2. **Feature engineering**
   - Experience converted to numeric (`<1` → 0, `>20` → 21)
   - Relevant experience → binary
   - Education level, company size and last new job → ordinal encoding
3. **Preprocessing pipeline**
   - `OneHotEncoder` for categorical features, `StandardScaler` for numerical features, wrapped in a `ColumnTransformer` + `Pipeline`
4. **Exploratory data analysis**
   - Target distribution, education, experience, training hours, city development index, correlation matrix
5. **Modeling** (80/20 stratified split)
   - Logistic Regression: `GridSearchCV`, 5-fold stratified CV, `class_weight='balanced'`
   - Random Forest: `RandomizedSearchCV`, 20 iterations, 3-fold stratified CV
   - Both tuned on F1-score
6. **Threshold tuning**
   - The decision threshold was selected on out-of-fold predictions from the training set to maximize F1 (chosen threshold for Random Forest: **0.54**)
7. **Candidate ranking**
   - Test-set candidates ranked by their predicted probability of switching

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7543 | 0.5052 | 0.7093 | 0.5901 | 0.7825 |
| **Random Forest** | **0.7988** | **0.5786** | **0.7104** | **0.6378** | **0.7994** |

**Random Forest is the final model.** It identifies about **7 out of 10** candidates who are open to a job change (recall 71%).

Because the classes are imbalanced, F1, recall and ROC-AUC are more informative here than accuracy alone.

![Confusion matrix - Random Forest](images/04_confusion_matrix_rf.png)

## 💡 Key Insights

- **City development index** is the strongest signal in the model (about 37% of the total feature importance), well ahead of training hours, experience and company size.
- Whether company information was missing also showed up among the top features.
- Feature importance shows what the model relies on, not what causes a job change.

![Feature importance](images/07_feature_importance.png)

## ⚠️ Limitations

- Precision is moderate (~58%), so some candidates flagged as likely to switch will not.
- There is a gap between train F1 (0.677) and test F1 (0.638), which suggests mild overfitting.
- Results come from a single public dataset and should not be used for real hiring decisions without further validation.

## 🚀 Future Improvements

- Try gradient boosting models (XGBoost / LightGBM) and resampling techniques
- Use permutation importance or SHAP for more reliable explanations
- Wrap the model in a simple web app for recruiters

## 🗂️ Repository Structure

```
├── Smart_Recruitment_Assistant.ipynb   # Full notebook: preparation, EDA, modeling, evaluation
├── processed_recruitment_data.csv      # Cleaned and engineered dataset
├── smart_recruitment_rf_model.pkl      # Trained Random Forest pipeline
├── images/                             # Charts exported from the notebook
└── README.md
```

## ▶️ How to Run

1. Download `aug_train.csv` from the *HR Analytics: Job Change of Data Scientists* dataset.
2. Open the notebook in Google Colab (or Jupyter) and update the path in the first data-loading cell.
3. Run all cells from top to bottom.

To load the trained model:

```python
import joblib

model = joblib.load('smart_recruitment_rf_model.pkl')
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.54).astype(int)
```

`X_new` must have the same columns as the training features (all columns except `target` and `city`).

## 👩‍💻 Author

**Yomna Saber**
AI & Data Science Enthusiast | Aspiring AI Engineer | Data Analyst

🔗 [LinkedIn](www.linkedin.com/in/yomnasaber)
