# Smart Recruitment Assistant

A small machine learning project that estimates how likely a candidate is to be open to a job change, so a recruiter can tell who is worth contacting first.

I built it during my AI summer training at the Information Technology Institute (ITI), using Python and scikit-learn.

## The idea

Only about 1 in 4 candidates in the data is actually looking to change jobs. If a recruiter contacts everyone, most of that effort goes to people who aren't interested. So instead of a yes/no answer, I wanted a probability for each candidate that I could use to rank them.

## The data

I used the public HR Analytics: Job Change of Data Scientists dataset from Kaggle. It has 19,158 rows and 13 columns, including city development index, education, experience, company size and type, training hours and how long ago the person last changed jobs. After removing 49 duplicate rows, 19,109 candidates were left.

The target is 1 if the person is looking for a job change and 0 if not. The split is roughly 25% / 75%.

![Target distribution](images/05_recruitment_statistics.png)

## What I did

**Cleaning.** I dropped the ID column and the duplicates. Missing values in gender, company size, company type and major were filled with "Unknown". For education, university enrollment, experience and last job change I used the most common value. I also added two flag columns showing where company size and company type were missing, because a missing value might carry some information too.

**Preparing the features.** Experience became a number, "has relevant experience" became 1/0, and education level, company size and last new job were turned into ordered scales. The remaining categories were one-hot encoded and the numeric columns were scaled, all inside one scikit-learn pipeline.

**Modeling.** I split the data 80/20 (stratified) and trained two models: Logistic Regression and Random Forest. Both were tuned for F1 with cross-validation, using grid search for Logistic Regression and random search for Random Forest.

**Threshold.** Instead of the default 0.5, I picked the decision threshold that gave the best F1 on the training data, so the test set stayed untouched. For the Random Forest it came out at 0.54.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7543 | 0.5052 | 0.7093 | 0.5901 | 0.7825 |
| Random Forest | 0.7988 | 0.5786 | 0.7104 | 0.6378 | 0.7994 |

Random Forest was better on every metric, and recall is almost the same for both. I went with the Random Forest as the final model.

In plain words: out of 10 candidates who really are open to a change, the model finds about 7. When it says someone will switch, it is right about 58% of the time.

Accuracy alone would be misleading here, since a model that always answers "no change" would already be right about 75% of the time. That is why I focused on recall, F1 and ROC-AUC.

![Confusion matrix](images/04_confusion_matrix_rf.png)

## What the model relies on

The city development index was by far the strongest feature (about 37% of the total importance), followed by training hours, experience and company size. Whether the company information was missing also made it into the top five.

This only shows what the model uses to make its predictions. It doesn't say why people change jobs.

![Feature importance](images/07_feature_importance.png)

## Limitations

- Precision is around 58%, so a fair number of the candidates it flags will not actually switch.
- Train F1 (0.677) is higher than test F1 (0.638), so there is a bit of overfitting.
- It's one public dataset. This shouldn't be used for real hiring decisions without more validation.

## Files

- `Smart_Recruitment_Assistant.ipynb`: the full notebook (cleaning, EDA, models, evaluation)
- `aug_train.csv`: the original dataset
- `processed_recruitment_data.csv`: the cleaned dataset
- `smart_recruitment_rf_model.pkl`: the trained Random Forest pipeline
- `images/`: charts exported from the notebook

## Running it

The original dataset is included in this repo as `aug_train.csv`, so nothing else needs to be downloaded.

1. Open `Smart_Recruitment_Assistant.ipynb` in Google Colab.
2. In the first data-loading cell, replace the file path with:

```python
df = pd.read_csv('https://raw.githubusercontent.com/yumnasaber/smart-recruitment-assistant/main/aug_train.csv')
```

3. Run all cells from top to bottom.

To use the saved model:

```python
import joblib

model = joblib.load('smart_recruitment_rf_model.pkl')
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.54).astype(int)
```

`X_new` needs the same columns as the training features (everything except `target` and `city`).

## Author

Yomna Saber
[LinkedIn](https://www.linkedin.com/in/yomnasaber)
