from src.data.DataMethods import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
import time
import pathlib
import matplotlib.pyplot as plt
import mlflow


def plot_model_details(metrics: list, logs: dict,save_path):
    count = 0
    
    for metric in metrics:
        models = list(logs.keys())
        values = [v[count] for v in logs.values()]

        plt.figure()
        bars = plt.bar(models, values)

        # ✅ ADD LABELS HERE
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )

        plt.title(f"Model {metric} Comparison")
        plt.xticks(rotation=30)
        plt.ylabel(metric)
        plt.tight_layout()

        file_name = f"{save_path}{metric}.png"
        plt.savefig(file_name)
        mlflow.log_artifact(file_name)

        count += 1
        
def plot_confusion_matrix(y_test,y_pred,model_name,save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix - {model_name}")

    file_name = f"{save_path}{model_name}_confusion_matrix.png"
    plt.savefig(file_name)

    mlflow.log_artifact(file_name)
    plt.close()

def plot_roc_curve(y_test, y_prob, model_name,save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')  # random line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()

    file_name = f"{save_path}{model_name}_roc_curve.png"
    plt.savefig(file_name)

    mlflow.log_artifact(file_name)

    plt.close()

def plot_precision_recall_curve(y_test, y_prob, model_name,save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name} (AP={ap:.3f})")

    file_name = f"{save_path}{model_name}_pr_curve.png"
    plt.savefig(file_name)

    mlflow.log_artifact(file_name)

    plt.close()

def plot_bubble(logs, save_path):
    models = list(logs.keys())

    f1 = [logs[m][1] for m in models]
    pred_time = [logs[m][5] for m in models]
    roc = [logs[m][3] for m in models]
    acc = [logs[m][0] for m in models]

    sizes = [r * 2000 for r in roc]  # bubble size

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(pred_time, f1, s=sizes, c=acc, cmap='viridis', alpha=0.7)

    for i, model in enumerate(models):
        plt.text(pred_time[i], f1[i], model)

    plt.xlabel("Prediction Time")
    plt.ylabel("F1 Score")
    plt.title("🚀 Model Performance Landscape")

    plt.colorbar(scatter, label="Accuracy")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}bubble_plot.png")


def main():
    print("start")
    corr_dir = pathlib.Path(__file__)
    home_dir = corr_dir.parent.parent.parent

    data_path = home_dir.as_posix() + "/data/processed/"
    output_path = home_dir.as_posix() + "/models/processed/"
    plot_path = home_dir.as_posix() + "/reports/figures/"

    print("mlflow load")
    # Specify the tracking URI for the MLflow server.
    mlflow.set_tracking_uri("http://localhost:5000")

    # Specify the experiment you just created for your LLM application or AI agent.
    mlflow.set_experiment("self_healing_model_v1")

    print("data loading")
    # loading the dataset
    train_data = DataLoader.load_data(data_path,"online_shoppers_intention_train.csv")
    test_data = DataLoader.load_data(data_path,"online_shoppers_intention_test.csv")

    X_train = train_data.drop(columns=["Revenue"])
    y_train = train_data["Revenue"]

    X_test = test_data.drop(columns=["Revenue"])
    y_test = test_data["Revenue"]

    models = {
                "RandomForest": RandomForestClassifier(random_state=42),
                "ExtraTrees": ExtraTreesClassifier(),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "AdaBoostClassifier":AdaBoostClassifier(),
                "xgboost":XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            }

    logs = {}

    metrics = ["Accuracy", "F1 Score", "Recall","ROC AUC", "Train_time","Pred_time"]

    print("model traing")
    for name, model in models.items():
        
        with mlflow.start_run(run_name=name):
            
            # Train
            start = time.time()
            model.fit(X_train, y_train) 
            train_time = time.time() - start
            # Predict
            start = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - start

            # Probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            recall = recall_score(y_test,y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)
            logs[name] = [acc,f1,recall,roc_auc, train_time, pred_time]
            
            # saving the plots
            plot_confusion_matrix(y_test,y_pred,name,plot_path)
            plot_roc_curve(y_test, y_prob, name,plot_path)
            plot_precision_recall_curve(y_test, y_prob, name,plot_path)
            plot_bubble(logs,plot_path)

            # Log metrics        
            mlflow.log_metric("accuracy", acc) # type: ignore
            mlflow.log_metric("f1_score", f1) # type: ignore
            mlflow.log_metric("recall",recall) # type: ignore
            mlflow.log_metric("roc_auc", roc_auc) # type: ignore
            mlflow.log_metric("Train_time",train_time)
            mlflow.log_metric("Pred_time",pred_time)


            
            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model") # type: ignore

    plot_model_details(metrics,logs,plot_path)
    mlflow.end_run()

if __name__ == "__main__":
    main()