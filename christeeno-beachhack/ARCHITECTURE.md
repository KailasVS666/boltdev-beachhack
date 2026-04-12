# AeroGuard System Architecture

This document outlines the data flow and architectural components of the AeroGuard system.

```mermaid
graph TD
    subgraph Data Source ["Data Ingestion Layer"]
        Sensors[Aircraft Sensors] -->|Raw Telemetry| API_In[Backend API Input]
        style Sensors fill:#f9f,stroke:#333,stroke-width:2px
    end

    subgraph Backend ["Backend Processing (Python/FastAPI)"]
        API_In --> Validator[Pydantic Validation]
        Validator --> Precalc[Feature Engineering]
        
        subgraph ML_Core ["ML Engine"]
            Precalc --> Scaler[Feature Scaler]
            Scaler --> IForest[Isolation Forest Model]
            IForest -->|Anomaly Score| Decision[Decision Logic]
            
            Context[NTSB Historical Data] -->|Narrative Context| Enrichment[Context Engine]
            Decision --> Enrichment
        end
        
        Enrichment --> Response[JSON Response Builder]
    end

    subgraph Frontend ["Presentation Layer (React/Vite)"]
        Poller[Data Polling Service] -->|GET /diagnosis| Response
        Response -->|JSON Data| State[Global State Store]
        
        State --> Dashboard[Main Dashboard]
        State --> Ghost[3D Ghost Model]
        
        Dashboard -->|Visualizes| Metrics[Gauges & Charts]
        Ghost -->|Highlights| Components[Faulty Components]
    end

    style Backend fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style Frontend fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style ML_Core fill:#fff9c4,stroke:#fbc02d,stroke-width:1px
```

## Data Flow Description

1.  **Ingestion**: Telemetry data (LATP, VEL, EGT, etc.) is received from the aircraft sensors (or simulation).
2.  **Processing**:
    *   **Validation**: `server_ml.py` ensures data integrity using Pydantic models.
    *   **ML Analysis**: Data is scaled and passed to the **Isolation Forest** to detect anomalies.
    *   **DQI Check**: Data Quality Index validates sensor reliability (detecting frozen/erroneous sensors).
3.  **Visualization**:
    *   The **React Frontend** polls the backend every 2 seconds.
    *   **Dashboard**: Displays real-time metrics (EGT, Vibration, Hydraulic Pressure).
    *   **3D Engine**: The Ghost Model highlights specific engine components in **Red/Yellow** based on the ML diagnosis.
