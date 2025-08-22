import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Fertilizer Recommendation API",
    description="An API to recommend fertilizer based on soil and crop data.",
    version="1.0.0"
)

# ➡️ Add this CORS middleware block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load Saved Artifacts ---
try:
    with open('Best_Fertilizer_Recommendation_Model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('fertilizer_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('fertilizer_model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('fertilizer_backend_mappings.pkl', 'rb') as f:
        backend_mappings = pickle.load(f)

    reverse_soil_map = {v: k for k, v in backend_mappings['soil_type_map'].items()}
    reverse_crop_map = {v: k for k, v in backend_mappings['crop_type_map'].items()}
    reverse_fertilizer_map = {v: k for k, v in backend_mappings['fertilizer_map'].items()}

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load a necessary artifact: {e}. Ensure all .pkl files are present.")

# --- 3. Define API Input/Output Models ---
class FertilizerInput(BaseModel):
    Temparature: int
    Humidity: int
    Moisture: int
    Soil_Type_ID: int
    Crop_Type_ID: int
    Nitrogen: int
    Potassium: int
    Phosphorous: int

    class Config:
        json_schema_extra = {
            "example": {
                "Temparature": 34,
                "Humidity": 65,
                "Moisture": 54,
                "Soil_Type_ID": 2,
                "Crop_Type_ID": 10,
                "Nitrogen": 38,
                "Potassium": 0,
                "Phosphorous": 0
            }
        }

# --- 4. API Endpoints ---
@app.get("/", summary="API Root", tags=["Status"])
def read_root():
    return {"message": "Welcome to the Fertilizer Recommendation API!"}

@app.get("/categories", summary="Get Soil and Crop Types", tags=["Categories"])
def get_categories():
    return {
        "soil_types": [{"id": v, "name": k} for k, v in backend_mappings['soil_type_map'].items()],
        "crop_types": [{"id": v, "name": k} for k, v in backend_mappings['crop_type_map'].items()]
    }

@app.post("/recommend", summary="Recommend Fertilizer", tags=["Prediction"])
def recommend_fertilizer(data: FertilizerInput):
    try:
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)
        input_df['Temparature'] = data.Temparature
        input_df['Humidity'] = data.Humidity
        input_df['Moisture'] = data.Moisture
        input_df['Nitrogen'] = data.Nitrogen
        input_df['Potassium'] = data.Potassium
        input_df['Phosphorous'] = data.Phosphorous

        soil_name = reverse_soil_map.get(data.Soil_Type_ID)
        crop_name = reverse_crop_map.get(data.Crop_Type_ID)

        if not soil_name or not crop_name:
            raise HTTPException(status_code=400, detail="Invalid Soil Type or Crop Type ID provided.")

        soil_col = f'Soil_Type_{soil_name}'
        crop_col = f'Crop_Type_{crop_name}'

        if soil_col in input_df.columns:
            input_df[soil_col] = 1
        if crop_col in input_df.columns:
            input_df[crop_col] = 1
        
        input_scaled = scaler.transform(input_df)
        predicted_fertilizer_num = model.predict(input_scaled)[0]
        recommendation = reverse_fertilizer_map.get(predicted_fertilizer_num)
        
        if not recommendation:
            raise HTTPException(status_code=500, detail="Could not map prediction to a fertilizer name.")

        return {"recommended_fertilizer": recommendation}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
