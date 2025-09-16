# 🔎 Job Change Prediction (HR Analytics)

Predict whether a candidate is likely to change jobs using machine learning.  
This project builds a **binary classification pipeline** on the Kaggle HR Analytics dataset.

---

## 🚀 What this project does
1. **Data loading**: Reads the dataset `aug_train.csv`.  
2. **Artificial missing values**: Uses `np.random.seed(0)` to randomly insert NaNs into numeric columns (to simulate real-world missingness).  
3. **ID removal**: Drops `enrollee_id` as it is not a predictive feature.  
4. **Exploration**: Inspects column types, class imbalance, and missing value counts.  
5. **Imputation**: Applies `SimpleImputer` to fill missing values (mean/median for numeric, most frequent for categorical).  
6. **Encoding**: Converts categorical features with `LabelEncoder` or `pd.get_dummies()`.  
7. **Scaling**: Standardizes numerical features with `StandardScaler`.  
8. **Train-test split**: Uses `train_test_split` with stratification to keep label balance.  
9. **Modeling**: Trains and compares:
10. **Cross-validation**: Evaluates models with `KFold` and `cross_val_score`.  
11. **Evaluation**: Reports `confusion_matrix` and `classification_report` (Accuracy, Precision, Recall, F1-score).  

---

## 📂 Repository structure
```
├── data/
│ └── aug_train.csv # dataset (to be uploaded)
├── job_change_prediction.ipynb # main notebook
└── README.md
```


---

## ▶️ Quick start
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/job-change-prediction.git
   cd job-change-prediction
   ```

2. Install dependencies:

   ```bash
pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. Launch Jupyter and run the notebook:

   ```bash
jupyter notebook job_change_prediction.ipynb
    ```

## 📊 Example workflow
- Run the notebook end-to-end to generate baseline models.

- Compare Logistic Regression, Random Forest, and SVM.

- Inspect confusion matrices and classification reports to evaluate performance.

## 🔧 Future improvements
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.

- Handle class imbalance with SMOTE or class weights.

- Add explainability with SHAP or LIME (not yet included).


