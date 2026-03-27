# 📅 Day 3 – Model Training, Evaluation & Experiment Tracking

## 🎯 Goal

Train multiple machine learning models, evaluate their performance using different metrics, and track experiments using MLflow.

---

## ✅ Tasks Completed

### 1. Model Training Script

Created model training script:

📁 `src/models/train_model.py`

Responsibilities:

* Train multiple machine learning models
* Evaluate model performance
* Generate visualizations
* Track experiments using MLflow

---

### 2. Models Implemented

Implemented multiple models for comparison:

```python
RandomForestClassifier
AdaBoostClassifier
ExtraTreesClassifier
XGBClassifier
```
👉 Enabled comparison of different algorithms on the same dataset.

---

### 3. Model Evaluation Metrics

Used multiple evaluation metrics:

```python
accuracy_score
f1_score
recall_score
roc_auc_score
average_precision_score
```
👉 Provided better understanding of model performance beyond accuracy.

---

### 4. Visualization Functions

Created reusable plotting functions:

```python
plot_confusion_matrix()
plot_roc_curve()
plot_precision_recall_curve()
plot_model_details()
plot_bubble()
```
Key Visualizations:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve
* Bubble Plot (Performance vs Time vs Accuracy)

👉 Provided better understanding of model performance beyond accuracy.

---

### 5. MLflow Integration

Integrated MLflow for experiment tracking:

``` python
import mlflow
```

Tracked:

* Model parameters
* Evaluation metrics
* Experiment runs

👉 Made experiments reproducible and easy to compare.

---

### 6. Experimentation

* Ran multiple experiments with different models
* Logged results using MLflow
* Compared models based on metrics and visualizations

👉 Enabled data-driven model selection.

---

🧠 Key Learnings


* Importance of evaluating models using multiple metrics
* Difference between Accuracy, F1 Score, and ROC-AUC
* How to compare models effectively
* Basics of MLflow for experiment tracking
*Importance of experiment logging in real-world ML systems

---

⚠️ Challenges Faced


* Understanding ROC Curve vs Precision-Recall Curve
* Managing outputs for multiple models
* Structuring MLflow logs properly
* Choosing the right evaluation metric

---

💬 Notes

* Accuracy alone is not reliable for evaluation
* MLflow helps in tracking and comparing experiments
* Visualization improves model understanding
* Running multiple models gives better insights

---

🚀 Next Plan (Day 4)

* Save best performing model
* Build prediction pipeline
* Create API using FastAPI
* Improve feature engineering
* Add monitoring (self-healing concept)