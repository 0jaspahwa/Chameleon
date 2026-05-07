import os
import json
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()

# Import your custom engine classes
from core import UltraPersonalizationEngine
from ml_models import DeepPredictor, UserSequenceLSTM, PersonalizationModel

app = FastAPI(title="Chameleon Personalization API")
engine = None

# Define the exact JSON structure the frontend will send us
class UserPayload(BaseModel):
    user_pseudo_id: str
    total_events: int = 1
    unique_pages: int = 1
    total_revenue: float = 0.0
    days_since_first_visit: int = 0
    recent_sequence: list[str] = []

@app.on_event("startup")
async def load_models():
    global engine
    print("🚀 Booting up API and loading ML models into memory...")
    
    asset_dir = "chameleon_api_assets"
    
    try:
        # 1. Initialize a "blank" engine (using a dummy dataframe)
        dummy_df = pd.DataFrame(columns=['user_pseudo_id', 'event_name', 'purchase_revenue', 'timestamp'])
        engine = UltraPersonalizationEngine(dummy_df, api_key=os.getenv("GEMINI_API_KEY"))
        
        # ⚠️ CRITICAL FIX: Give the engine dummy tables WITH columns so pandas doesn't crash!
        engine.user_features = pd.DataFrame(columns=['user_pseudo_id', 'target_cluster', 'total_revenue', 'total_events'])
        engine.merged_df = pd.DataFrame(columns=['user_pseudo_id', 'ItemCategory', 'ItemName', 'event_name', 'timestamp'])
        
        # 2. Overwrite the blank ML pipeline with your saved weights
        engine.pm = PersonalizationModel(rf_threshold=0.65, lstm_threshold=0.70)
        
        # Load Scikit-Learn assets
        engine.pm.kmeans = joblib.load(f"{asset_dir}/kmeans.joblib")
        engine.pm.kmeans_scaler = joblib.load(f"{asset_dir}/kmeans_scaler.joblib")
        engine.pm.feature_scaler = joblib.load(f"{asset_dir}/feature_scaler.joblib")
        
        if os.path.exists(f"{asset_dir}/random_forest.joblib"):
            engine.pm.rf = joblib.load(f"{asset_dir}/random_forest.joblib")
            
        # Load Vocab & Features
        with open(f"{asset_dir}/event2id.json", "r") as f:
            engine.pm.event2id = json.load(f)
        with open(f"{asset_dir}/feature_cols.json", "r") as f:
            engine.pm.feature_cols = json.load(f)
            
        # Load PyTorch Deep Model
        engine.pm.deep = DeepPredictor(input_size=len(engine.pm.feature_cols) + 1, num_classes=engine.pm.kmeans.n_clusters)
        engine.pm.deep.load_state_dict(torch.load(f"{asset_dir}/deep_model.pth", map_location=torch.device('cpu'), weights_only=True))
        engine.pm.deep.eval()
        
        # Load PyTorch Sequence Model
        engine.pm.sequence_model = UserSequenceLSTM(vocab_size=len(engine.pm.event2id), num_classes=engine.pm.kmeans.n_clusters, pad_idx=0)
        engine.pm.sequence_model.load_state_dict(torch.load(f"{asset_dir}/lstm_model.pth", map_location=torch.device('cpu'), weights_only=True))
        engine.pm.sequence_model.eval()

        print("✅ Models successfully loaded from disk. Ready for traffic!")
        
    except Exception as e:
        print(f"❌ Critical Error loading models: {e}")

@app.post("/api/v1/personalize")
async def personalize_ui(payload: UserPayload):
    if not engine or getattr(engine.pm, 'kmeans', None) is None:
        raise HTTPException(status_code=503, detail="Models are still loading or offline...")
        
    # Convert payload to dictionary for your core.py logic
    profile_dict = payload.model_dump() if hasattr(payload, 'model_dump') else payload.dict()
    
    try:
        # Prevent engine from trying to look up history in the dummy dataframe
        profile_dict['recent_sequence'] = payload.recent_sequence
        
        result = engine.generate_personalized_landing_page(profile_dict)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))