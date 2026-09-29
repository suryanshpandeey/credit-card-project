# 🛒 Big Mart Sales Prediction

A machine learning regression project that predicts **item-level sales at Big Mart outlets** using product and outlet characteristics.

This project focuses on building a **leakage-free machine learning workflow**, including model-based missing-value imputation, validation-based model selection, and final evaluation on an untouched test set.

---

## 📌 Project Overview

The goal of this project is to predict:

```text
Item_Outlet_Sales
```

using information about the item and the outlet where it is sold.

The project compares:

* Linear Regression
* Random Forest Regression
* Multiple Random Forest hyperparameter configurations

A key focus is ensuring that **information from the validation and test sets is never used to fit preprocessing steps or select the final model**.

---

## 🎯 Objectives

* Explore the Big Mart sales dataset
* Identify and handle missing values
* Clean inconsistent categorical values
* Engineer an additional item category feature
* Split the dataset into train, validation, and test sets
* Perform leakage-free missing-value imputation
* Encode categorical variables
* Compare Linear Regression and Random Forest
* Tune Random Forest hyperparameters
* Select the best model using validation performance
* Evaluate the selected model on the test set

---

# 📊 Dataset

The dataset contains:

* **8,523 observations**
* **12 original columns**
* Numerical and categorical features
* `Item_Outlet_Sales` as the target variable

### Features

| Feature                     | Description                            |
| --------------------------- | -------------------------------------- |
| `Item_Identifier`           | Unique product identifier              |
| `Item_Weight`               | Weight of the product                  |
| `Item_Fat_Content`          | Fat-content category                   |
| `Item_Visibility`           | Visibility of the product in the store |
| `Item_Type`                 | Product type                           |
| `Item_MRP`                  | Maximum Retail Price                   |
| `Outlet_Identifier`         | Unique outlet identifier               |
| `Outlet_Establishment_Year` | Year the outlet was established        |
| `Outlet_Size`               | Size of the outlet                     |
| `Outlet_Location_Type`      | Location tier of the outlet            |
| `Outlet_Type`               | Type of outlet                         |
| `Item_Outlet_Sales`         | **Target variable**                    |

---

# 🔍 Missing Values

The original dataset contains missing values in two columns:

| Column        | Missing Values |
| ------------- | -------------: |
| `Item_Weight` |          1,463 |
| `Outlet_Size` |          2,410 |

No missing values are present in the other columns.

Instead of simply applying mean/mode imputation before splitting the data, the project handles these missing values using information learned from the **training set only**.

---

# 🧹 Data Cleaning

## Item Fat Content

The dataset contains inconsistent representations of the same categories.

The following values are standardized:

```text
low fat → Low Fat
LF      → Low Fat
reg     → Regular
```

This produces two consistent categories:

```text
Low Fat
Regular
```

---

## Item Category Feature

A new feature is extracted from `Item_Identifier`:

```python
Item_Category = Item_Identifier.str[:2]
```

The first two characters represent broad product categories:

```text
FD → Food
DR → Drinks
NC → Non-consumable
```

This feature is used during the `Item_Weight` imputation process.

---

# 🔀 Train / Validation / Test Split

The dataset is split **before any learned preprocessing**.

```text
60% Training
20% Validation
20% Testing
```

The resulting sizes are:

```text
Training:   5,113 rows
Validation: 1,705 rows
Testing:    1,705 rows
```

The split uses:

```python
random_state = 42
```

This separation ensures that the validation and test datasets do not influence the preprocessing models or hyperparameter selection.

---

# 🛡️ Leakage-Free Preprocessing

The central idea of the project is:

> **Split first, learn preprocessing only from training data, then transform validation and test data.**

The workflow is:

```text
                Raw Dataset
                     │
                     ▼
              Row-wise Cleaning
                     │
                     ▼
             Train / Val / Test
              60% / 20% / 20%
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Item Weight              Outlet Size
    Imputation               Imputation
          │                     │
          └──────────┬──────────┘
                     ▼
                   EDA
                     │
                     ▼
             Feature Encoding
                     │
                     ▼
            Model Development
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Linear Regression      Random Forest
          │                     │
          └──────────┬──────────┘
                     ▼
            Validation Selection
                     │
                     ▼
               Final Model
                     │
                     ▼
              Test Evaluation
```

The notebook explicitly states that the imputation models, encoders, and sales model are fitted using the training split.

---

# 🧩 Missing Value Imputation

## 1. Item Weight Imputation

Missing `Item_Weight` values are predicted using a separate machine learning model.

The model uses:

```text
Item_Identifier
Item_Fat_Content
Item_Type
Item_Category
Item_MRP
Item_Visibility
```

The target of this sub-problem is:

```text
Item_Weight
```

The project compares:

* Linear Regression
* Random Forest Regression with multiple hyperparameter combinations

The known-weight training observations are further split into training and validation sets for selecting the imputation model.

### Selected Model

The best model for predicting missing `Item_Weight` values was:

**Linear Regression**

Validation performance:

```text
RMSE = 1.5785
R²   = 0.8832
```

The selected model is then used to fill missing weights in the train, validation, and test datasets.

---

## 2. Outlet Size Imputation

Missing `Outlet_Size` values are imputed using the most common outlet size for each `Outlet_Type`.

The mapping is learned using the **training data only**:

| Outlet Type       | Imputed Size |
| ----------------- | ------------ |
| Grocery Store     | Small        |
| Supermarket Type1 | Small        |
| Supermarket Type2 | Medium       |
| Supermarket Type3 | Medium       |

After imputation, all three datasets contain zero missing values in `Outlet_Size`.

---

# 📊 Exploratory Data Analysis

EDA is performed on the training data after missing-value imputation.

The notebook examines:

* Descriptive statistics
* Numerical feature distributions
* Item weight
* Item visibility
* Item MRP
* Outlet sales
* Categorical variables
* Relationships between predictors and sales

Histograms with KDE are generated for:

```text
Item_Weight
Item_Visibility
Item_MRP
Item_Outlet_Sales
```

The training set has an average `Item_Outlet_Sales` of approximately **2,220.46**.

---

# ⚙️ Feature Preprocessing

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Numerical features are passed through without transformation.

A `ColumnTransformer` is used to combine these transformations, and the preprocessing is placed inside a Scikit-learn `Pipeline`.

This ensures that the encoder is fitted only on the relevant training data.

---

# 🤖 Machine Learning Models

## Linear Regression

Linear Regression is used as a baseline regression model.

The model is combined with the preprocessing pipeline:

```text
Categorical Features
        │
        ▼
One-Hot Encoding
        │
        ▼
Linear Regression
```

---

## Random Forest Regression

Random Forest Regression is used to capture nonlinear relationships between the item/outlet characteristics and sales.

Multiple hyperparameter combinations are evaluated.

Parameters explored include:

```text
n_estimators
max_depth
min_samples_leaf
max_features
```

Each candidate model is trained on the training set and evaluated on the validation set.

---

# 📏 Evaluation Metrics

Two regression metrics are used.

## RMSE

**Root Mean Squared Error**

```text
RMSE = √MSE
```

Lower values indicate smaller prediction errors.

## R² Score

R² measures the proportion of variation in the target explained by the model.

Higher values indicate better explanatory performance.

The model-selection process primarily ranks candidates using **validation RMSE**.

---

# 🏆 Model Selection

The best Random Forest configuration on the validation set was:

```text
n_estimators = 300
max_depth = 8
```

with the remaining hyperparameters determined by the searched configuration.

### Validation Performance

| Model             | Validation RMSE | Validation R² |
| ----------------- | --------------: | ------------: |
| **Random Forest** |     **1032.51** |    **0.6103** |
| Linear Regression |         1074.68 |        0.5778 |

The selected Random Forest therefore had a validation RMSE of **1032.51** and R² of **0.6103**.

---

# 📈 Final Model Performance

The selected Random Forest model is evaluated on all three splits.

| Dataset    |         R² |        RMSE |
| ---------- | ---------: | ----------: |
| Training   |     0.6615 |     1005.08 |
| Validation |     0.6103 |     1032.51 |
| **Test**   | **0.5986** | **1071.24** |

The final test performance is:

```text
R²   = 0.5986
RMSE = 1071.24
```

---

# 🔬 Results Interpretation

The final model explains approximately **59.86% of the variance** in item-level outlet sales on the test set, based on the reported R².

The test RMSE is approximately **1071.24 sales units**, meaning the model's prediction errors have a substantial magnitude relative to the sales distribution.

The validation and test results are relatively close:

```text
Validation RMSE: 1032.51
Test RMSE:       1071.24
```

which provides a useful indication of how the selected model generalizes to the held-out test data.

---

# 🧠 Key Takeaways

### 1. Avoiding data leakage

The dataset is split before learned imputation and encoding.

### 2. Model-based imputation

Missing `Item_Weight` is treated as a supervised regression problem rather than being replaced with a simple mean.

### 3. Group-based imputation

`Outlet_Size` is imputed using outlet-type-specific modes calculated from training data.

### 4. Validation-based model selection

The validation set is used to select the best sales prediction model and Random Forest configuration.

### 5. Untouched test evaluation

The test set is reserved for final performance evaluation.

### 6. Random Forest outperforms the Linear Regression baseline

The selected Random Forest achieves lower validation RMSE and higher validation R² than Linear Regression.

---

# 🗂️ Project Structure

```text
Big-Mart-Sales-Prediction/
│
├── Project_12_Big_Mart_Sales_Prediction_fixed.ipynb
├── sale/
│   └── Train.csv
│
└── README.md
```

The notebook expects the dataset at:

```python
DATA_PATH = 'sale/Train.csv'
```

Update this path if the dataset is stored elsewhere.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

### Machine Learning

* Linear Regression
* Random Forest Regression
* One-Hot Encoding
* Model-based imputation
* Hyperparameter search
* Train/validation/test methodology

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/big-mart-sales-prediction.git
cd big-mart-sales-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Add the dataset

Place the dataset at:

```text
sale/Train.csv
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

```text
Project_12_Big_Mart_Sales_Prediction_fixed.ipynb
```

Run the cells sequentially.

---

# 📌 Project Summary

```text
Dataset              : Big Mart Sales
Observations         : 8,523
Train / Val / Test   : 60% / 20% / 20%
Random Seed          : 42

Missing Item Weight  : 1,463
Missing Outlet Size  : 2,410

Weight Imputation    : Linear Regression
Weight RMSE          : 1.5785
Weight R²            : 0.8832

Sales Models         : Linear Regression / Random Forest
Best Sales Model     : Random Forest
Validation RMSE      : 1032.51
Validation R²        : 0.6103

Test RMSE            : 1071.24
Test R²              : 0.5986
```

---

## 👤 Author

**Suryansh Pandey**

If you found this project useful, consider ⭐ starring the repository.
