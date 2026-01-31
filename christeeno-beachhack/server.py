"""
AeroGuard FastAPI Backend Server
Connects Python ML models to React frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.aeroguard_engine import AeroGuardEngine

# Initialize FastAPI
app = FastAPI(
    title="AeroGuard API",
    description="Predictive Maintenance Intelligence for Aviation",
    version="1.0.0"
)

# CORS - Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[AeroGuardEngine] = None


# Request/Response Models
class SensorFrame(BaseModel):
    """Single frame of sensor data from frontend"""
    LATP: float
    LONP: float
    ALT: float
    VEL: float
    TH: float
    N1: float
    N2: float
    EGT: float
    FF: float
    VIB: float
    VRTG: float
    OIL_P: float
    OIL_T: float
    FLAP: float
    HYDY: float


class FeedbackRequest(BaseModel):
    """Technician feedback on alert"""
    alert_id: str
    accepted: bool
    reason: Optional[str] = None
    technician_id: Optional[str] = None


# Startup Event
@app.on_event("startup")
async def startup_event():
    """Load ML models on server startup"""
    global engine
    try:
        print("🚀 Loading AeroGuard ML Engine...")
        engine = AeroGuardEngine()
        print("✅ AeroGuard Engine Ready")
    except Exception as e:
        print(f"❌ Failed to load engine: {e}")
        print("⚠️  Server will run but predictions will fail")


# Health Check
@app.get("/health")
async def health_check():
    """Server health status"""
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "version": "1.0.0"
    }


# Main Diagnosis Endpoint (Frontend polls this)
@app.get("/diagnosis")
async def get_diagnosis():
    """
    Main endpoint that frontend polls every 2 seconds.
    Returns current system diagnosis with mock data for now.
    Once we have real streaming data, this will analyze actual sensor readings.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="ML Engine not loaded")
    
    # For now, return a safe "monitoring" response
    # In production, this would analyze real-time sensor streams
    return {
        "overall_status": "MONITORING",
        "timestamp": "2026-01-31T14:15:00Z",
        "diagnosis_details": {
            "root_cause_diagnosis": "All systems nominal",
            "evidence_summary": "Continuous monitoring active. No anomalies detected.",
            "maintenance_action": {
                "amm_reference": "N/A",
                "priority": "ROUTINE"
            }
        },
        "confidence": 0.95
    }


# Real-Time Prediction Endpoint
@app.post("/predict")
async def predict_failure(frame: SensorFrame):
    """
    Analyze a single sensor frame and return prediction.
    This is the core ML prediction endpoint.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="ML Engine not loaded")
    
    try:
        # Convert Pydantic model to dict
        frame_data = frame.dict()
        
        # Run through AeroGuard engine
        report = engine.analyze_frame(frame_data)
        
        return {
            "status": "success",
            "prediction": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Feedback Endpoint
@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Technician feedback on alerts.
    Feeds into the One-Way Valve feedback loop.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="ML Engine not loaded")
    
    # Log feedback
    print(f"📝 Feedback received: Alert {feedback.alert_id} - {'ACCEPTED' if feedback.accepted else 'REJECTED'}")
    
    # In production, this would trigger model retraining pipeline
    return {
        "status": "feedback_recorded",
        "alert_id": feedback.alert_id,
        "message": "Feedback submitted successfully"
    }


# Alerts History
@app.get("/alerts")
async def get_alerts():
    """Get recent alert history"""
    # This would query a database in production
    return {
        "alerts": [],
        "count": 0
    }


if __name__ == "__main__":
    print("🛡️  AeroGuard Backend Server")
    print("📡 Starting on http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
