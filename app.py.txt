from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import pickle

app = FastAPI()

# Load trained model (make sure this file exists — use pickle or joblib)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the input data structure
class Product(BaseModel):
    our_price: float
    competitor_price: float
    stock: int
    demand_index: float
    category: str  # Should be one of: Electronics, Clothing, Home, Books

# One-hot encode category manually (you could automate this)
def preprocess_input(data: Product):
    row = {
        'our_price': data.our_price,
        'competitor_price': data.competitor_price,
        'stock': data.stock,
        'demand_index': data.demand_index,
        'category_Electronics': 1 if data.category == "Electronics" else 0,
        'category_Clothing': 1 if data.category == "Clothing" else 0,
        'category_Home': 1 if data.category == "Home" else 0,
        'category_Books': 1 if data.category == "Books" else 0,
    }
    return pd.DataFrame([row])

@app.post("/recommend_price/")
def recommend_price(product: Product):
    df = preprocess_input(product)
    prediction = model.predict(df)[0]
    return {
        "recommended_price": round(prediction, 2)
    }
