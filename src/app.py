from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from src.predict import load_model, predict

app= FastAPI(title="Churn Prediction API",version="1.0.0")

model=load_model("models/XGBoost_model.pkl")

class CustomerData(BaseModel):
    Account_length: int
    Area_code:int
    International_plan:int
    Voice_mail_plan:int 
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls:int
    Total_eve_minutes:float
    Total_eve_calls:int
    Total_night_minutes:float
    Total_night_calls:int
    Total_intl_minutes:float
    Total_intl_calls:int
    Customer_service_calls:int

@app.get("/health")
def health():
    return {"Status" : "API IS RUNNING!!!"}

@app.post("/predict")

def get_predictions(data : CustomerData):
    input_data = {
        "Account length": data.Account_length,
        "Area code"	:data.Area_code,
        "International plan":data.International_plan,	
        "Voice mail plan"	: data.Voice_mail_plan,
        "Number vmail messages":data.Number_vmail_messages,
        "Total day minutes"	:data.Total_day_minutes,
        "Total day calls"	: data.Total_day_calls,
        "Total eve minutes"	: data.Total_eve_minutes,
        "Total eve calls"	: data.Total_eve_calls,
        "Total night minutes"	: data.Total_night_minutes,
        "Total night calls"	: data.Total_night_calls,
        "Total intl minutes"	: data.Total_intl_minutes,
        "Total intl calls"	: data.Total_intl_calls,
        "Customer service calls": data.Customer_service_calls
    }

    result = predict(model,input_data)
    return result