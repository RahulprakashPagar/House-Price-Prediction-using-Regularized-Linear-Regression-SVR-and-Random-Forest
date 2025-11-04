# House-Price-Prediction-using-Regularized-Linear-Regression-SVR-and-Random-Forest
A machine learning project that predicts house prices using multiple regression techniques — Linear Regression (with L1, L2, and Elastic Net regularization), Support Vector Regression, and Random Forest Regression — applied on the KC Housing dataset from Kaggle.

# 🏠 House Price Prediction using Regularized Regression and Ensemble Models

This project applies **Supervised Machine Learning (Regression)** techniques to predict housing prices using the **KC Housing Dataset** from Kaggle.  
It explores **L1 (Lasso), L2 (Ridge), Elastic Net, Support Vector Regression (SVR)**, and **Random Forest Regression**, comparing their performance, interpretability, and overfitting behavior.

---

## 📊 Project Overview
- **Goal:** Predict house prices (`price`) based on housing features  
- **Dataset:** [KC Housing Dataset on Kaggle](https://www.kaggle.com/datasets?search=kc_)  
- **Techniques Used:** Linear Regression (L1/L2/Elastic Net), SVR, Random Forest  
- **Libraries:** pandas, sklearn, matplotlib, plotly  
- **Environment:** Google Colab / Jupyter Notebook  
- **Author:** [Rahul Pagar](https://www.linkedin.com/in/rahul-pagar1993)

---

### 🏷️ Keywords
machine-learning, regression, linear-regression, elastic-net, ridge, lasso, support-vector-regression, random-forest, python, sklearn, data-science

---

## ⚙️ Workflow and Methods

### 1️⃣ Data Preparation
- Imported dataset using `read_csv()`
- Dropped unnecessary columns (e.g., `date`)
- Checked for missing values → none found
- Created correlation matrix and **heatmap** to identify multicollinearity  
- Dropped highly correlated feature (`sqft_above`)
- Standardized features using `StandardScaler()`

---

### 2️⃣ Linear Regression Models

#### • Without Regularization
- Implemented using `SGDRegressor(penalty=None)`
- Tuned `eta0` (learning rate) and `max_iter` using GridSearchCV  
- **Best R² = 0.6997** → ~70% variance explained

#### • Elastic Net Regularization (L1 + L2)
- Combines Lasso and Ridge effects for balanced sparsity + shrinkage  
- Tuned `alpha`, `eta0`, `l1_ratio`, `max_iter` with GridSearchCV  
- **Optimal Params:** `alpha=0.01`, `eta0=0.001`, `l1_ratio=0.40`, `max_iter=1500`  
- **R² = 0.6998**

#### • Ridge Regression (L2)
- Penalizes large coefficients, reduces overfitting  
- **R² = 0.6993**

#### • Lasso Regression (L1)
- Performs feature selection by shrinking coefficients to zero  
- **R² = 0.6997**

---

### 3️⃣ Support Vector Regression (SVR)

#### • Without Regularization
- Kernels = [`linear`, `poly`, `rbf`, `sigmoid`]; Epsilon = [100, 1000, 10000]  
- **R² = 0.11 (underfitting)**

#### • With Regularization (L2)
- Tuned `kernel`, `C`, `epsilon` using GridSearchCV (CV = 10)  
- **Optimal Params:** `kernel='poly'`, `C=10000`, `epsilon=1000`  
- **R² = 0.76 (76%)**

---

### 4️⃣ Random Forest Regression
- Tuned `n_estimators` and `max_features` via GridSearchCV (CV = 5)
- **Optimal Params:** `n_estimators=24`, `max_features=None`  
- **R² = 0.9798 (97.98%)** → Indicates possible overfitting

#### • Feature Importance
Top features:  
`grade`, `sqft_living`, `lat`, `long`, `waterfront`

After selecting important features and re-training:  
**R² = 0.9761 (97.61%)**

---

## 🧠 Model Comparison

| Model | Regularization | R² Score | Interpretability | Notes |
|-------|----------------|----------|------------------|-------|
| Linear Regression | None | 0.6997 | ✅ High | Baseline |
| Elastic Net | L1+L2 | 0.6998 | ✅ Balanced | Best among linear |
| Lasso | L1 | 0.6997 | ✅ High (sparse model) | Feature selection |
| Ridge | L2 | 0.6993 | ✅ Medium | No feature removal |
| SVR (with L2) | L2 (C = 10000) | 0.76 | ⚙️ Medium | Handles non-linear patterns |
| Random Forest | Tree ensemble | 0.9761 | ❌ Low | Overfitting observed |

---

## 📈 Visualizations
- Correlation Heatmap (Plotly)  
- Beta Coefficient Bar Charts (for L1, L2, Elastic Net)  
- Feature Importance Bar Chart (Random Forest)

---

## 📑 Files, Author & Conclusion

### 📁 Files Included
- **`Code_Rahul Pagar.py`** — Full Python implementation  
- **`Data File_kc_housing Data.csv`** — Kaggle dataset  
- **`Report_Rahul_Pagar.pdf`** — Comprehensive report & analysis  

---

### 👨‍💻 Author
**Rahul Pagar**  
🎓 MSc in Business Analytics — Dublin Business School  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/rahul-pagar1993)

---

### 🏁 Conclusion
Among all models, **Random Forest** achieved the highest R² (97.98%) but showed overfitting.  
**Elastic Net** and **Lasso regression** offer a better balance between accuracy, interpretability, and generalization.  
**Support Vector Regression (SVR)** performed moderately well for non-linear patterns.
