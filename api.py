from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import logging
import os
import time
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Guardian AI v8.0",
    description="AI-Powered Fraud Detection System",
    version="8.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model
import pickle
model = None
model_path = os.environ.get("MODEL_PATH", "/app/models/guardian_model.pkl")

def load_model():
    global model
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model not loaded: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "8.0.0",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
  
class TransactionRequest(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float
    merchant_id: str
    user_id: str
    merchant_category: str = "general"
    channel: str = "online"
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class FraudResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: str
    action: str
    reasons: List[str]
    processing_time_ms: float

def get_risk_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def get_action(score: float) -> str:
    if score >= 0.8:
        return "BLOCK"
    elif score >= 0.6:
        return "CHALLENGE"
    elif score >= 0.4:
        return "REVIEW"
    else:
        return "ALLOW"

@app.post("/analyze", response_model=FraudResponse)
async def analyze_transaction(request: TransactionRequest):
    start_time = time.time()
    try:
        import numpy as np
        features = [
            request.amount,
            hash(request.merchant_id) % 1000,
            hash(request.user_id) % 1000,
            hash(request.merchant_category) % 100,
            1 if request.channel == "online" else 0,
        ]
        if model is not None:
            score = float(model.predict_proba([features])[0][1])
        else:
            score = min(request.amount / 100000, 0.99)
        
        processing_time = (time.time() - start_time) * 1000
        return FraudResponse(
            transaction_id=request.transaction_id,
            fraud_score=round(score, 4),
            is_fraud=score >= 0.5,
            risk_level=get_risk_level(score),
            action=get_action(score),
            reasons=["ML model analysis", "Amount analysis"],
            processing_time_ms=round(processing_time, 2)
        )
    logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "Guardian AI v8.0",
        "status": "running",
        "endpoints": ["/health", "/analyze", "/docs"]
    }

@app.get("/stats")
async def get_stats():
    return {
        "total_transactions": 0,
        "fraud_detected": 0,
        "model_version": "8.0.0",
        "uptime": "running"
    }

@app.post("/batch-analyze")
async def batch_analyze(transactions: List[TransactionRequest]):
    results = []
    for txn in transactions:
        result = await analyze_transaction(txn)
        results.append(result)
    return {"results": results, "total": len(results)}

@app.get("/model/info")
async def model_info():
    return {
        "model_loaded": model is not None,
        "model_path": model_path,
        "version": "8.0.0"
    }
