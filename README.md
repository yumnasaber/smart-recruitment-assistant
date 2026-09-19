# Smart Recruitment Assistant 🔎

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-gray?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)

> A machine learning project that estimates how likely a candidate is to be open to a job change, helping recruiters prioritize who is worth contacting first.
>
> *Built during my AI summer training at the Information Technology Institute (ITI).*

## 📌 The Problem

Only about **1 in 4** candidates in the data is actually looking to change jobs. If a recruiter contacts everyone, most of that effort goes to people who aren't interested. Instead of a simple yes/no answer, this project calculates a probability for each candidate so recruiters can rank and target them effectively.

## 📊 The Dataset

I used the public [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists) dataset from Kaggle.

- **Size:** 19,158 rows and 14 columns in the raw file (19,109 rows after removing 49 duplicates).
- **Features included:** City development index, education, experience, company size/type, training hours, and last job change.
- **Target:** `1` (looking for a job change) vs `0` (not looking).
- **Distribution:** Highly imbalanced (roughly 25% / 75%).

![Target distribution](images/05_recruitment_statistics.png)

## ⚙️ Approach & Workflow

* **Data Cleaning:** Dropped the ID column and duplicates. Missing values in gender, company size, company type, and major were filled with `"Unknown"`. For education, university enrollment, experience, and last job change, I used the most common value. I also created two flag columns to track missing company size/type, as the "missingness" itself might carry useful information.
* **Feature Engineering:** Converted experience to numeric, flagged relevant experience as binary (1/0), and mapped education level, company size, and last new job into ordinal scales.
* **Pipeline:** Remaining categorical variables were one-hot encoded and numeric columns were scaled, all bundled inside a single `scikit-learn` pipeline.
* **Modeling:** Split the data 80/20 (stratified) and trained two models: **Logistic Regression** and **Random Forest**. Both were tuned for F1 score using cross-validation (Grid Search for LR, Random Search for RF).
* **Threshold Tuning:** Instead of the default 0.5, I selected the decision threshold that maximized the F1 score on the *training data* (using cross-validated predictions), so the threshold itself was not tuned on the test set. For the Random Forest, this optimal threshold was **0.54**.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7543 | 0.5052 | 0.7093 | 0.5901 | 0.7825 |
| **Random Forest (Final)** | **0.7988** | **0.5786** | **0.7104** | **0.6378** | **0.7994** |

Because of the imbalanced target, accuracy alone is misleading (a model predicting "no change" every time would already reach about 75%). This is why I focused on **Recall, F1, and ROC-AUC**.

The Random Forest performed better on every metric. In plain words: **out of 10 candidates who really are open to a change, the model identifies about 7.** When it flags someone as likely to switch, it is correct roughly 58% of the time.

![Confusion matrix](images/04_confusion_matrix_rf.png)

## 🔍 Feature Importance

The **city development index** was by far the strongest feature (about 37% of the total importance), followed by training hours, experience, and company size. Whether the company information was missing also made it into the top five features.

*Note: this only shows what the model relies on to make its predictions. It doesn't explain why people actually change jobs.*

![Feature importance](images/07_feature_importance.png)

## ⚠️ Limitations

- **Precision is around 58%**, so a fair number of the candidates flagged by the model will not actually switch.
- **Slight overfitting:** Train F1 (0.677) is higher than Test F1 (0.638).
- **Scope:** It's trained on one public dataset and shouldn't be used for real hiring decisions without further validation.

## 🔮 Future Improvements

- **Try stronger models:** Compare the Random Forest against gradient boosting models such as XGBoost or LightGBM.
- **Compare ways of handling imbalance:** The models currently rely on class weights. Resampling techniques like SMOTE could be tested against them.
- **Move imputation into the pipeline:** Missing values were filled with the most common value before the train/test split. Doing it inside the pipeline would remove that small source of leakage.
- **Use the `city` column:** I dropped it because it has many unique values. Frequency or target encoding might make it usable.
- **Better explanations:** Use permutation importance or SHAP, since the impurity-based importance used here tends to favour continuous features.
- **Cleaner model selection:** Choose between models on a separate validation set and calibrate the predicted probabilities.
- **A simple app:** Wrap the model in a small Streamlit app where a recruiter can enter a candidate profile and get a probability back.

## 📂 Repository Structure

- `Smart_Recruitment_Assistant.ipynb`: the full notebook (cleaning, EDA, models, evaluation)
- `aug_train.csv`: the original Kaggle dataset
- `processed_recruitment_data.csv`: the cleaned dataset
- `smart_recruitment_rf_model.pkl`: the trained Random Forest pipeline
- `images/`: charts exported from the notebook

## 🚀 How to Run

The original dataset is included in this repo (`aug_train.csv`), so you don't need to download anything else.

1. Open `Smart_Recruitment_Assistant.ipynb` in [Google Colab](https://colab.research.google.com/).
2. In the first data-loading cell, replace the local file path with the raw GitHub link:

```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/yumnasaber/smart-recruitment-assistant/main/aug_train.csv')
```

3. Run all cells from top to bottom.

**To use the saved model on new data:**

```python
import joblib

# Load the trained pipeline
model = joblib.load('smart_recruitment_rf_model.pkl')

# Get probabilities for the positive class
probabilities = model.predict_proba(X_new)[:, 1]

# Apply the tuned threshold of 0.54
predictions = (probabilities >= 0.54).astype(int)
```

*`X_new` needs the same columns as the training features (everything except `target` and `city`).*

---

**Author:** Yomna Saber | [LinkedIn](https://www.linkedin.com/in/yomnasaber)
