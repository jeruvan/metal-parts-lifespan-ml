# Metal Parts Lifespan — Regression & Classification (COMP1801)

Two-task ML project for a manufacturing dataset:
1) **Regression** — predict lifetime (hours) of metal parts.
2) **Classification** — label parts into **Low / Medium / High** lifespan groups from K-Means, then build a supervised model.

## 🔧 Tech & Methods
- **Pipeline**: `ColumnTransformer(StandardScaler + OneHotEncoder)` → model
- **Split**: 70/15/15 (train/val/test)
- **Imbalance**: `SMOTE` (classification)
- **Models**: RandomForest, MLPRegressor / MLPClassifier
- **Tuning**: `RandomizedSearchCV` / `GridSearchCV`
- **Explainability**: Feature importances, SHAP (for NN)

## 📊 Results (highlights)
**Regression (test set):**
- Random Forest (tuned): **R² ≈ 0.910**, **RMSE ≈ 96.58**, **MAE ≈ 74.57**  
- MLP Regressor (tuned): **R² ≈ 0.820**, **RMSE ≈ 138.82**, **MAE ≈ 108.11**

**Classification (test set):**
- Random Forest: **Accuracy ≈ 0.847**  
- MLP Classifier: **Accuracy ≈ 0.807**

> Key drivers: **coolingRate**, **castType**, **seedLocation**; non-linear interactions present.

## 📂 Structure
```
comp1801-metal-parts-lifespan-ml/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ COMP1801.ipynb
├─ src/                 # (optional future: modular code)
├─ data/
│  └─ README.md
├─ assets/              # export charts for README
└─ reports/
   ├─ COMP1801_Colab_Notebook.pdf
   ├─ COMP1801_Coursework_1.pdf
   └─ COMP1801_Coursework_2.pdf
```

## ▶️ Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Open `notebooks/COMP1801.ipynb`, set the dataset path, and run.

## ▶️ Run in Google Colab
- Upload the notebook.
- Install deps at the top cell:
  ```python
  !pip install -q scikit-learn imbalanced-learn shap seaborn
  ```
- Mount/import your dataset CSV, then run all cells.


## 🔍 Notes
- K-Means used to derive **Low/Medium/High** labels; only the **High** cluster mostly meets the >1500h requirement (some edge cases just below threshold).  
- Random Forest recommended for classification (balanced accuracy, interpretability).  
- For regression, Random Forest achieved the strongest test metrics; NN chosen for interpretability of categorical signals in some reports.

## ✅ Next improvements
- Calibrated classification thresholds to enforce the 1500h rule.
- Add **permutation importance** + SHAP for both tasks.
- Track experiments with a seed and a fixed train/val/test split artifact.
