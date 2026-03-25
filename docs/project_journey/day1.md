# 📅 Day 1 – Project Setup & Data Pipeline Foundation

## 🎯 Goal

Set up the foundation of the self-healing ML project with proper structure, version control, and reusable data loading system.

---

## ✅ Tasks Completed

### 1. Project Initialization

* Initialized Git repository:

```bash
git init
```

* Initialized DVC for data version control:

```bash
dvc init
```

👉 This will help in tracking datasets and models separately from Git.

---

### 2. Project Structure Setup

Created a clean and scalable project structure:

```

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


```

👉 This follows industry-standard ML project organization.

---

### 3. Dataset Added

* Added dataset:

```
online_shoppers_intention.csv
```

* Stored inside:

```
data/raw/
```

---

### 4. DataLoader Class Implementation

Created a reusable data handling class:

📁 `src/data/DataMethods.py`

Key Features:

* Load CSV data
* Save processed data
* Error handling included

```python
class DataLoader:

    @staticmethod
    def load_data(path: str, name: str):
        data = pd.read_csv(path + name)
        return data

    @staticmethod
    def save_data(path: str, name: str, data):
        data.to_csv(os.path.join(path, name), index=False)
```

👉 This allows reuse across:

* data ingestion
* feature engineering
* model training

---

### 5. Data Ingestion Script

Created entry script:

📁 `src/data/make_dataset.py`

Responsibilities:

* Locate project root
* Load dataset using DataLoader
* Verify data shape

```python
data = DataLoader.load_data(data_path, "online_shoppers_intention.csv")
print(data.shape)
```

✅ Output:

```
(12330, 18)
```

---

## 🧠 Key Learnings

* Importance of project structure in ML systems
* Difference between script-based vs modular coding
* Handling file paths using `pathlib`
* Avoiding global variables → using reusable classes
* Basics of DVC for data versioning

---

## 🚀 Next Plan (Day 2)

* Build data preprocessing pipeline
* Handle missing values
* Feature encoding
* Save processed dataset to `data/processed/`

---

## 💬 Notes

* Faced path issues initially → resolved using proper path handling
* Learned correct way to run modules using:

```bash
python -m src.data.make_dataset
```

---

🔥 Day 1 Status: **Strong Foundation Built**
