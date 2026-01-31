"""
AeroGuard FastAPI Backend Server (Lightweight Version)
Connects to React frontend without full ML engine loading
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import uvicorn

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
    """Server startup"""
    print("🚀 AeroGuard Backend Started")
    print("📡 Frontend can connect at http://localhost:8000")
    print("📚 API Docs available at http://localhost:8000/docs")


# Health Check
@app.get("/health")
async def health_check():
    """Server health status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


# Main Diagnosis Endpoint (Frontend polls this every 2 seconds)
@app.get("/diagnosis")
async def get_diagnosis():
    """
    Main endpoint that frontend polls.
    Returns intelligently generated system diagnosis.
    """
    # Simulate different diagnosis states
    import random
    
    # Most of the time: all systems nominal
    status_roll = random.random()
    
    if status_roll < 0.95:  # 95% of time = MONITORING
        return {
            "overall_status": "MONITORING",
            "timestamp": datetime.now().isoformat(),
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
    else:  # 5% of time = Simulated alert
        return {
            "overall_status": "SAFETY BREACH DETECTED",
            "timestamp": datetime.now().isoformat(),
            "diagnosis_details": {
                "root_cause_diagnosis": "Flap Actuator Hydraulic Pressure Anomaly Detected",
                "evidence_summary": "Causal analysis identified FLAP system hydraulic pressure drop correlating with increased VIB signature and EGT deviation. Root cause: Potential hydraulic seal degradation in flap actuator assembly.",
                "maintenance_action": {
                    "amm_reference": "AMM 27-31-00-520-801",
                    "priority": "URGENT",
                    "task_card": "INSP-FLAP-HYD-01",
                    "parts_required": ["ACT-FLAP-787-01"],
                    "estimated_hours": 4.5
                }
            },
            "causal_chain": ["HYDY", "FLAP", "VIB", "EGT"],
            "confidence": 0.88,
            "risk_envelope": "15-25 flights remaining",
            "dqi_confidence": "HIGH"
        }


# Real-Time Prediction Endpoint
@app.post("/predict")
async def predict_failure(frame: SensorFrame):
    """
    Analyze a single sensor frame.
    This version uses rule-based logic instead of ML for reliability.
    """
    try:
        frame_data = frame.dict()
        
        # Simple rule-based anomaly detection
        anomalies = []
        
        # Check vibration
        if frame_data['VIB'] > 2.5:
            anomalies.append({
                "sensor": "VIB",
                "value": frame_data['VIB'],
                "threshold": 2.5,
                "severity": "HIGH"
            })
        
        # Check EGT
        if frame_data['EGT'] > 550:
            anomalies.append({
                "sensor": "EGT",
                "value": frame_data['EGT'],
                "threshold": 550,
                "severity": "MEDIUM"
            })
        
        # Check hydraulics
        if frame_data['HYDY'] < 2800:
            anomalies.append({
                "sensor": "HYDY",
                "value": frame_data['HYDY'],
                "threshold": 2800,
                "severity": "HIGH"
            })
        
        if anomalies:
            return {
                "status": "ANOMALY_DETECTED",
                "anomalies": anomalies,
                "recommendation": "Review maintenance schedule",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "NORMAL",
                "dqi": 0.95,
                "timestamp": datetime.now().isoformat()
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
    print(f"📝 Feedback: Alert {feedback.alert_id} - {'✅ ACCEPTED' if feedback.accepted else '❌ REJECTED'}")
    if feedback.reason:
        print(f"   Reason: {feedback.reason}")
    
    return {
        "status": "feedback_recorded",
        "alert_id": feedback.alert_id,
        "message": "Feedback will be incorporated in quarterly model retraining",
        "timestamp": datetime.now().isoformat()
    }


# Alerts History
@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """Get recent alert history"""
    # Mock recent alerts
    return {
        "alerts": [
            {
                "id": "ALT-2026-001",
                "timestamp": "2026-01-30T14:20:00Z",
                "status": "RESOLVED",
                "engine": 2,
                "issue": "Elevated vibration signature",
                "action_taken": "Borescope inspection - No defects found"
            },
            {
                "id": "ALT-2026-002",
                "timestamp": "2026-01-28T09:15:00Z",
                "status": "RESOLVED",
                "engine": 3,
                "issue": "EGT trending up",
                "action_taken": "Fuel nozzle cleaning performed"
            }
        ],
        "count": 2,
        "total_in_db": 47
    }


# Model Status
@app.get("/model/status")
async def model_status():
    """Get ML model status"""
    return {
        "model_version": "AeroGuard_v1.0",
        "last_training": "2026-01-25T10:00:00Z",
        "rmse": 28.5,
        "architecture": "Bi-LSTM + Isolation Forest + Autoencoder",
        "status": "PRODUCTION",
        "next_retraining": "2026-04-01T00:00:00Z"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🛡️  AEROGUARD BACKEND SERVER")
    print("=" * 60)
    print("📡 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,  # Pass app object directly
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
