# Titanic Survival Prediction API

A machine learning API that predicts Titanic passenger survival, built with Scikit-learn and served via FastAPI.

## What It Does
Takes passenger details (class, age, sex, fare, etc.) and predicts survival using a trained Logistic Regression model.

## Tech Stack
- Python, Pandas, Scikit-learn
- FastAPI + Uvicorn
- Joblib (model persistence)
- Postman (API testing)

## Model Performance
- **Accuracy: 81.01%** (vs. ~62% baseline of always guessing "did not survive")
- Trained on the Kaggle Titanic dataset (891 passengers)

## Project Structure

    ml-fastapi-project/
    ├── data/train.csv          # Raw dataset
    ├── 01_explore_data.ipynb   # Data exploration
    ├── 02_train_model.ipynb    # Cleaning, encoding, training
    ├── main.py                 # FastAPI app
    ├── titanic_model.pkl       # Saved trained model
    └── requirements.txt

## How to Run
1. Clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`
6. Visit `http://127.0.0.1:8000/docs` to test interactively

## Example Request
POST `/predict`

    {
      "Pclass": 1,
      "Sex": 1,
      "Age": 29,
      "SibSp": 0,
      "Parch": 0,
      "Fare": 100,
      "Embarked_Q": 0,
      "Embarked_S": 1
    }

## Response

    {
      "survived": 1,
      "message": "Survived"
    }