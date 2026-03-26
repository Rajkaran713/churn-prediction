import mlflow
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, precision_score

def evaluate_model(model, X_Test, Y_Test):

    with mlflow.start_run(run_name="final_xgboost_evaluation"):
        y_pred=model.predict(X_Test)

        metrics=({
            'test_accuracy': accuracy_score(Y_Test,y_pred),
            'test_recall': recall_score(Y_Test,y_pred),
            'test_precision': precision_score(Y_Test,y_pred),
            'test_f1_score': f1_score(Y_Test,y_pred)
        })

        mlflow.log_metrics(metrics)

        #print classification report 
        print("CLASSIFICATION REPORT:")
        print(classification_report(Y_Test,y_pred))

        print("METRICS:")
        print(f"Accuracy score: {metrics['test_accuracy']:.4f}")
        print(f"Recall Score : {metrics['test_recall']:.4f}")
        print(f"F1_score: {metrics['test_f1_score']:.4f}")
        print(f"Precision score: {metrics['test_precision']:.4f}")

    return metrics