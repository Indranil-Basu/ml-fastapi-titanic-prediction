from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('titanic_model.pkl')

class Passenger(BaseModel):
    Pclass: int
    Sex : int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int
    Embarked_S: int
    
@app.post("/predict")
def predict_survival(passenger: Passenger):
    input_data = np.array([[
        
        passenger.Pclass,
        passenger.Sex,
        passenger.Age,
        passenger.SibSp,
        passenger.Parch,
        passenger.Fare,
        passenger.Embarked_Q,
        passenger.Embarked_S
    ]])    
    
    prediction = model.predict(input_data)[0]
    
    return {
        "survived" : int(prediction),
        "message": "Survived" if prediction ==1 else "Did not Survive"
        }