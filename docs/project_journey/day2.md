# 📅 Day 2 – Feature Engineering & Preprocessing Pipeline

## 🎯 Goal

Build a reusable preprocessing system to clean, transform, and prepare data for machine learning while preventing data leakage.

---

## ✅ Tasks Completed

### 1. Feature Engineering Script

Created feature engineering script:

📁 `src/features/build_features.py`

Responsibilities:

* Load train & test datasets
* Apply transformations
* Scale features
* Save processed data

---

### 2. Train-Test Data Handling

* Loaded separate datasets:

```python
train_data = DataLoader.load_data(data_path,
"online_shoppers_intention_train.csv")
test_data = DataLoader.load_data(data_path, "online_shoppers_intention_test.csv")

```

### 3. Handling Missing Values

* Calculated median from training data:
``` python
median_val = train_data["ProductRelated_Duration"].median()
```

``` python
train_data = column_transformation(train_data, median_val)
test_data = column_transformation(test_data, median_val)
```

* Prevented data leakage by not using test data statistics.

### 4. Column Transformation Function

* Created reusable function:

``` python
def column_transformation(data: pd.DataFrame, median_val) -> pd.DataFrame:
```

Key Responsibilities:

* Handle missing values
* Apply basic preprocessing
* Return transformed dataframe

👉 Made preprocessing reusable across pipeline stages.

### 5. Feature Scaling & Encoding

``` python
def column_scaling(train: pd.DataFrame, test: pd.DataFrame):
```

Used:

* StandardScaler → Numerical features
* OneHotEncoder → Categorical features
* ColumnTransformer → Combined pipeline

👉 Ensured consistent transformations across train and test datasets.

### 6. Saving Processed Data

Saved processed datasets:

``` python
DataLoader.save_data(output_path, "online_shoppers_intention_train.csv", train)
DataLoader.save_data(output_path, "online_shoppers_intention_test.csv", test)
```

👉 Created clean processed datasets for model training.

🧠 Key Learnings
* Importance of avoiding data leakage
* Why preprocessing must use only training data statistics
* How to structure reusable transformation functions
* Basics of ColumnTransformer for combining multiple 
* transformations
* Difference between raw vs processed data layers

⚠️ Challenges Faced
* Handling missing values for only one column (needs generalization)
* Pipeline not fully automated yet
* Confusion around when to split vs preprocess (resolved: split first, then preprocess)
* Need better structure for feature engineering

💬 Notes
* Always compute statistics (median/mean) using training data only
* Apply same transformation to test data
* Functions help in making pipeline reusable and clean