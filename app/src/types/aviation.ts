// NASA DASHlink Data Types
export interface FlightDataRow {
  // Time
  GMT_HOUR: number;
  GMT_MINUTE: number;
  GMT_SEC: number;

  // Environment
  ALT: number;        // Altitude in feet
  SAT: number;        // Static Air Temperature (Celsius)
  GS: number;         // Ground Speed
  VRTG: number;       // Vertical G-load
  LATG: number;       // Lateral G-load

  // Hydraulics
  HYDY: number;       // Yellow System Pressure (0-3000 psi)
  HYDG: number;       // Green System Pressure (0-3000 psi)

  // Control Surfaces
  ROLL: number;       // Roll angle (-180 to 180 deg)
  PTCH: number;       // Pitch angle (-30 to 30 deg)
  RUDD: number;       // Rudder position (-30 to 30 deg)
  AIL_1: number;      // Aileron 1 position
  AIL_2: number;      // Aileron 2 position
  ELEV_1: number;     // Elevator 1 position
  ELEV_2: number;     // Elevator 2 position

  // Engine 1 Data
  EGT_1: number;      // Exhaust Gas Temperature
  N1_1: number;       // Fan Speed
  N2_1: number;       // Core Speed
  FF_1: number;       // Fuel Flow
  VIB_1: number;      // Vibration
  OIL_TEMP_1: number; // Oil Temperature
  OIL_PRESS_1: number;// Oil Pressure

  // Engine 2 Data
  EGT_2: number;
  N1_2: number;
  N2_2: number;
  FF_2: number;
  VIB_2: number;
  OIL_TEMP_2: number;
  OIL_PRESS_2: number;

  // Engine 3 Data
  EGT_3: number;
  N1_3: number;
  N2_3: number;
  FF_3: number;
  VIB_3: number;
  OIL_TEMP_3: number;
  OIL_PRESS_3: number;

  // Engine 4 Data
  EGT_4: number;
  N1_4: number;
  N2_4: number;
  FF_4: number;
  VIB_4: number;
  OIL_TEMP_4: number;
  OIL_PRESS_4: number;

  // AI/ML Fields
  RUL_SCORE: number;  // Remaining Useful Life Score
  ANOMALY_FLAG: number; // 0 = normal, 1 = anomaly detected
}

export interface EngineData {
  id: number;
  egt: number;
  n1: number;
  n2: number;
  ff: number;
  vib: number;
  oilTemp: number;
  oilPress: number;
  historicalMeanEGT: number;
  rulScore: number;
}

// Error categories for system-wide diagnostics
export type ErrorCategory =
  // Existing categories
  | 'ENGINE_OVERHEAT'    // EGT threshold exceeded
  | 'ENGINE_FAN_SPOOL'   // N1/N2 out of range
  | 'VIBRATION'          // VIB threshold exceeded
  | 'OIL_FUEL_SYSTEM'    // Oil pressure/fuel flow issues
  | 'STRUCTURAL'         // VRTG (vertical G-load) issues
  | 'HYDRAULICS'         // HYDY/HYDG pressure issues
  | 'FLIGHT_CONTROLS'    // Flaps, ailerons, rudder issues
  | 'AVIONICS'           // GPS/throttle/cockpit systems
  // New error signatures
  | 'ENGINE_THERMAL_FATIGUE'    // EGT > 820°C
  | 'BEARING_VIBRATION'         // VIB > 2.5
  | 'HYDRAULIC_LEAK'            // HYDY/HYDG < 2600 psi
  | 'STRUCTURAL_STRESS'         // VRTG > 1.85G
  | 'FUEL_FEED_INCONSISTENCY'   // FF drift > 10%
  | 'FLIGHT_CONTROL_LAG'        // FLAP/RUDD error > 2°
  | 'AILERON_FAULT'             // AIL lag > 0.5s
  | 'PITOT_STATIC_BLOCKAGE';    // ALT/CAS inconsistency

export type ErrorPriority = 'CRITICAL' | 'URGENT' | 'WATCHLIST';

export interface SystemError {
  category: ErrorCategory;
  severity: 'warning' | 'critical';  // Keep for backward compatibility
  priority?: ErrorPriority;          // New priority system
  affectedMeshes: string[];
  message: string;
  value?: number;
  threshold?: number;
  ammReference?: string;             // AMM task reference
}

export interface AircraftSystemHealth {
  // Engine metrics (already in EngineData)
  engines: EngineData[];

  // Structural
  vrtg: number;           // Vertical G-load
  latg: number;           // Lateral G-load

  // Hydraulics
  hydy: number;           // Yellow system pressure (psi)
  hydg: number;           // Green system pressure (psi)

  // Flight controls
  flaps: number;          // Flap position
  ailerons: number;       // Aileron position
  rudder: number;         // Rudder position
  elevator: number;       // Elevator position

  // Avionics
  gpsStatus: 'nominal' | 'warning' | 'critical';
  throttleStatus: 'nominal' | 'warning' | 'critical';

  // Active errors
  activeErrors: SystemError[];
}

export interface SystemState {
  status: 'nominal' | 'alert' | 'crisis';
  message: string;
  affectedEngine?: number;
}

export interface XAIInsight {
  failureProbability: number;
  primaryCause: string;
  explanation: string;
  featureImportance: {
    feature: string;
    responsibility: number;
  }[];
}

export interface MaintenanceAction {
  ammReference: string;
  partNumber: string;
  partName: string;
  stockCount: number;
  estimatedCost: number;
}

export type ViewMode = 'dashboard' | 'diagnostic';
