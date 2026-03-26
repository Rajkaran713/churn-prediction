import mlflow
from preprocess import data_loading_1, data_loading_2,data_concatenation, data_splitting,preprocess
from train import train_final_xgboost, train_random_forest, train_xgboost_baseline
from evaluate import evaluate_model

def main():
    mlflow.set_experiment(experiment_name="Churn-Prediction")

    print("Data Loading")
    df1=data_loading_1("data/churn-bigml-20.csv")
    df2= data_loading_2("data/churn-bigml-80.csv")

    print("Data Concatenation")
    df=data_concatenation(df1,df2)

    print("Data Preprocessing")
    df=preprocess(df)

    print("Data Splitting")
    X_Train,X_Test,Y_Train,Y_Test=data_splitting(df)

    print("Training Random Forest")
    rf_model=train_random_forest(X_Train,Y_Train)

    print("Training XGBoost Baseline Model")
    xgb_baseline=train_xgboost_baseline(X_Train,Y_Train)

    print("Training XGBoost Final Model with MLFLOW tracking")
    xgb_final=train_final_xgboost(X_Train,Y_Train)

    print("Final Model Evaluation")
    evaluate_model(xgb_final,X_Test,Y_Test)

    print("\nPIPELINE COMPLETED SUCESSFULLY!")

if __name__ == "__main__":
    main()
