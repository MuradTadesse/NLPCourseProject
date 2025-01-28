# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.deployment.model_server import ModelServer
from src.deployment.inference import InferenceEngine
from config.config import MODEL_CONFIG
import os
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

app = FastAPI()

class Query(BaseModel):
    text: str

# Initialize the model server and inference engine globally
def initialize_model():
    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / 'data' / 'models' / 'dashen_bot_model.pt'
        
        if not model_path.exists():
            print("Model file not found. Training new model...")
            from src.train import main as train_model
            train_model()
        
        model_server = ModelServer(
            model_path=str(model_path),
            tokenizer_name=MODEL_CONFIG['model_name']
        )
        
        return InferenceEngine(model_server, {})
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return None

inference_engine = initialize_model()

@app.post("/query")
async def query_bot(query: Query):
    if not inference_engine:
        raise HTTPException(
            status_code=500,
            detail="Model initialization failed. Please check server logs."
        )
    
    try:
        response = inference_engine.get_response(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if inference_engine:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}