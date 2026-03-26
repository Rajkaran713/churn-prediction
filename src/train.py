import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score, classification_report, f1_score,precision_score,precision_recall_curve
import pickle
import os

def train_random_forest(X_Train, Y_Train):
    rf=RandomForestClassifier(
        n_estimators=300,
        criterion='gini',
        max_features='sqrt',
        random_state=42,
        oob_score=True,
        class_weight='balanced'
    )
    rf.fit(X_Train,Y_Train)
    return rf

def train_xgboost_baseline(X_Train, Y_Train):
    xgb_model=xgb.XGBClassifier(
        random_state=42,
        scale_pos_weight=5.8
    )
    xgb_model.fit(X_Train,Y_Train)
    return xgb_model

def train_final_xgboost(X_Train, Y_Train):
    best_params = {
        'subsample': 1,
        'scale_pos_weight': 6,
        'reg_lambda': 0,
        'reg_alpha': 0,
        'n_estimators': 100,
        'min_child_weight': 4,
        'max_depth': 5,
        'learning_rate': 0.01,
        'gamma': 0.3,
        'colsample_bytree': 0.7,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    with mlflow.start_run(run_name="final_xgboost"):
        
        #Log parameters
        mlflow.log_params(best_params)
        
        #train model
        model=xgb.XGBClassifier(**best_params)
        model.fit(X_Train, Y_Train)

        #evaluate
        y_pred=model.predict(X_Train)
        metrics={
            'train_accuracy': accuracy_score(Y_Train,y_pred),
            'train_recall': recall_score(Y_Train,y_pred),
            'train_f1':f1_score(Y_Train,y_pred),
            'train_precision':precision_score(Y_Train,y_pred)
        }

        #log metrics
        mlflow.log_metrics(metrics)

        #save model
        os.makedirs("models",exist_ok=True)
        model_path="models/XGboost_model.pkl"
        with open(model_path,'wb') as f:
            pickle.dump(model,f)

        #log model artifact
        mlflow.log_artifact(model_path)

    return model

