import pandas as pd
import pickle

def load_model(model_path="models/XGBoost_model.pkl"):
    with open(model_path,'rb') as f:
        model=pickle.load(f)
    return model

def predict(model, input_data: dict):
    df= pd.DataFrame([input_data])
    prediction= model.predict(df)
    probabilities=model.predict_proba(df)[0][1]

    return {
        "Churn prediction": int(prediction[0]),
        "Churn Probability" : round(float(probabilities),4),
        "Churn Label": "Yes" if prediction[0]==1 else "No"
    }