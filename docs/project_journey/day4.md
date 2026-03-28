# 📅 Day 4 – Advanced Model Optimization & Selection (Optuna + MLflow)

---

## 🎯 Goal

Enhance model performance using **hyperparameter tuning** and build a robust system to select the **best model automatically**.

---

## ✅ Tasks Completed

### 🔹 1. Multi-Model Training & Evaluation

* Trained multiple models:

  * RandomForest
  * ExtraTrees
  * LogisticRegression
  * AdaBoost
  * XGBoost
* Evaluated using:

  * Accuracy
  * F1 Score
  * Recall
  * ROC-AUC

---

### 🔹 2. Top Model Selection

* Sorted models based on **ROC-AUC**
* Selected **Top 3 models** for further optimization

---

### 🔹 3. Hyperparameter Tuning using Optuna

* Implemented automated tuning using Optuna
* Created separate optimization functions for each model:

  * RandomForest
  * ExtraTrees
  * XGBoost
  * LogisticRegression
  * AdaBoost
* Used:

  * Cross-validation (cv=3)
  * ROC-AUC as optimization metric
* Ran multiple trials to find best parameters

---

### 🔹 4. MLflow Integration (Advanced Tracking)

* Used MLflow for experiment tracking
* Logged:

  * Hyperparameters (per trial)
  * ROC-AUC scores
  * Best parameters
  * Best score
* Implemented **nested runs** for each Optuna trial

---

### 🔹 5. Tuned Model Evaluation

* Retrained models using best parameters
* Evaluated on test data
* Compared tuned models

---

### 🔹 6. Final Model Selection

* Selected final best model based on tuned ROC-AUC
* Stored:

  * Model object
  * Performance score

---

### 🔹 7. Model Persistence

* Saved final model using:

```python
joblib.dump(final_model, "models/final_model.joblib")
```

---

## 📊 Key Insights

* Hyperparameter tuning improved model performance slightly but consistently
* Tree-based models (RandomForest / ExtraTrees / XGBoost) performed best
* Optuna efficiently converged to optimal parameters
* MLflow provided excellent visibility into experiments and trials
* Selecting top models before tuning is more efficient than tuning all models

---

## ⚠️ Challenges Faced

* Confusion between MLflow `log_model` vs `load_model`
* Handling Optuna trial logging properly
* Managing multiple models and tuning workflows
* Small improvements despite tuning (model saturation)
* Increased computation time due to multiple trials

---

## 🧠 Learnings

* Hyperparameter tuning is refinement, not magic improvement
* Data quality matters more than model tuning
* Separation of training and inference pipelines is critical
* Experiment tracking is essential for real-world ML systems
* Smart model selection > brute-force tuning

---

## 🚀 Next Plan (Day 5)

* Build prediction pipeline (`predict_model.py`)
* Integrate model with API using FastAPI
* Implement monitoring system
* Add self-healing mechanism (auto retraining)
* Improve feature engineering

---

## 💬 Summary

Today marked a major upgrade in the project:

> From simple model training → to a full ML optimization pipeline with automated tuning and tracking.

This is a crucial step toward building a **production-ready self-healing ML system**.

---
