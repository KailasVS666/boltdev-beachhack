import type { FlightDataRow, EngineData, XAIInsight, MaintenanceAction } from '@/types/aviation';

// Generate realistic NASA DASHlink-style flight data
export function generateMockFlightData(): FlightDataRow[] {
  const data: FlightDataRow[] = [];
  const baseTime = new Date('2024-01-15T08:00:00Z');

  // Historical means for EGT (Exhaust Gas Temperature)
  const historicalMeans = {
    EGT_1: 520,
    EGT_2: 525,
    EGT_3: 518,
    EGT_4: 522
  };

  for (let i = 0; i < 500; i++) {
    const timeOffset = i * 500; // 500ms intervals
    const currentTime = new Date(baseTime.getTime() + timeOffset);

    // Simulate flight phases
    let altitude: number;
    let phase: 'ground' | 'climb' | 'cruise' | 'descent';

    if (i < 50) {
      phase = 'ground';
      altitude = 0;
    } else if (i < 150) {
      phase = 'climb';
      altitude = (i - 50) * 350;
    } else if (i < 350) {
      phase = 'cruise';
      altitude = 35000 + Math.sin(i * 0.1) * 500;
    } else {
      phase = 'descent';
      altitude = Math.max(0, 35000 - (i - 350) * 300);
    }

    // === MULTI-SYSTEM ERROR SIMULATION ===
    // Different error types trigger at different row ranges to showcase all 8 categories

    // 1. ENGINE OVERHEAT (EGT) - Starts at row 150
    const egtAnomaly = i > 150 ? (i - 150) * 0.8 : 0;

    // 2. ENGINE FAN/SPOOL (N1/N2) - Starts at row 200
    const fanSpoolAnomaly = i > 200 ? (i - 200) * 0.15 : 0;

    // 3. VIBRATION - Starts at row 300 (existing)
    const engine2Vibration = i > 300 ? 1.5 + (i - 300) * 0.02 : 1.0;

    // 4. OIL/FUEL SYSTEM - Starts at row 250
    const oilPressureDrop = i > 250 ? (i - 250) * 0.08 : 0;

    // 5. STRUCTURAL (VRTG) - Severe turbulence at row 280-310
    const structuralStress = (i > 280 && i < 310) ? Math.sin((i - 280) * 0.3) * 1.5 : 0;

    // 6. HYDRAULICS - Pressure drop at row 320
    const hydraulicDegradation = i > 320 ? (i - 320) * 8 : 0;

    // 7. FLIGHT CONTROLS - Anomaly at row 350
    const controlSurfaceAnomaly = i > 350 ? (i - 350) * 0.5 : 0;

    // 8. AVIONICS - GPS/Throttle issues at row 380
    const avionicsFlag = i > 380 ? 1 : 0;

    // Simulate flight dynamics for control surfaces
    const baseRoll = Math.sin(i * 0.02) * 5; // Gentle banking
    const basePitch = phase === 'climb' ? 8 : phase === 'descent' ? -5 : 2;
    const turbulence = i > 250 && i < 280 ? Math.random() * 3 : 0; // Turbulence zone

    const row: FlightDataRow = {
      GMT_HOUR: currentTime.getUTCHours(),
      GMT_MINUTE: currentTime.getUTCMinutes(),
      GMT_SEC: currentTime.getUTCSeconds(),

      ALT: Math.round(altitude),
      SAT: Math.round(15 - altitude / 1000 * 2),
      GS: phase === 'cruise' ? 450 + Math.random() * 20 : 200 + Math.random() * 100,
      VRTG: 1.0 + Math.random() * 0.1 + turbulence * 0.3 + structuralStress,
      LATG: Math.random() * 0.1 - 0.05 + turbulence * 0.1,

      // Hydraulics (nominal: 2800-3000 psi) - DEGRADING
      HYDY: Math.round(Math.max(1500, 2900 - hydraulicDegradation + Math.random() * 100 - 50)),
      HYDG: Math.round(Math.max(1500, 2950 - hydraulicDegradation + Math.random() * 80 - 40)),

      // Control Surfaces - WITH ANOMALIES
      ROLL: Math.round((baseRoll + Math.random() * 2 - 1) * 10) / 10,
      PTCH: Math.round((basePitch + Math.random() * 2 - 1) * 10) / 10,
      RUDD: Math.round((Math.sin(i * 0.015) * 3 + controlSurfaceAnomaly + Math.random() * 1) * 10) / 10,
      AIL_1: Math.round((baseRoll * 0.5 + controlSurfaceAnomaly * 0.3 + Math.random() * 2) * 10) / 10,
      AIL_2: Math.round((-baseRoll * 0.5 + controlSurfaceAnomaly * 0.3 + Math.random() * 2) * 10) / 10,
      ELEV_1: Math.round((basePitch * 0.3 + controlSurfaceAnomaly * 0.2 + Math.random() * 1) * 10) / 10,
      ELEV_2: Math.round((basePitch * 0.3 + controlSurfaceAnomaly * 0.2 + Math.random() * 1) * 10) / 10,

      // Engine 1 - Normal
      EGT_1: Math.round(historicalMeans.EGT_1 + Math.random() * 20 - 10),
      N1_1: Math.round(85 + Math.random() * 5),
      N2_1: Math.round(92 + Math.random() * 3),
      FF_1: Math.round(2500 + Math.random() * 200),
      VIB_1: 1.0 + Math.random() * 0.3,
      OIL_TEMP_1: Math.round(85 + Math.random() * 10),
      OIL_PRESS_1: Math.round(45 + Math.random() * 5),

      // Engine 2 - MULTIPLE ANOMALIES (EGT, N1/N2, VIB, OIL)
      EGT_2: Math.round(historicalMeans.EGT_2 + egtAnomaly + Math.random() * 20 - 10),
      N1_2: Math.round(Math.max(15, 85 - fanSpoolAnomaly + Math.random() * 5)),
      N2_2: Math.round(Math.max(20, 92 - fanSpoolAnomaly * 0.8 + Math.random() * 3)),
      FF_2: Math.round(Math.max(500, 2500 - oilPressureDrop * 10 + Math.random() * 200)),
      VIB_2: Math.min(4.0, engine2Vibration + Math.random() * 0.2),
      OIL_TEMP_2: Math.round(85 + egtAnomaly * 0.1 + Math.random() * 10),
      OIL_PRESS_2: Math.round(Math.max(10, 45 - oilPressureDrop + Math.random() * 5)),

      // Engine 3 - Normal
      EGT_3: Math.round(historicalMeans.EGT_3 + Math.random() * 20 - 10),
      N1_3: Math.round(85 + Math.random() * 5),
      N2_3: Math.round(92 + Math.random() * 3),
      FF_3: Math.round(2500 + Math.random() * 200),
      VIB_3: 1.0 + Math.random() * 0.3,
      OIL_TEMP_3: Math.round(85 + Math.random() * 10),
      OIL_PRESS_3: Math.round(45 + Math.random() * 5),

      // Engine 4 - Normal
      EGT_4: Math.round(historicalMeans.EGT_4 + Math.random() * 20 - 10),
      N1_4: Math.round(85 + Math.random() * 5),
      N2_4: Math.round(92 + Math.random() * 3),
      FF_4: Math.round(2500 + Math.random() * 200),
      VIB_4: 1.0 + Math.random() * 0.3,
      OIL_TEMP_4: Math.round(85 + Math.random() * 10),
      OIL_PRESS_4: Math.round(45 + Math.random() * 5),

      // AI/ML predictions
      RUL_SCORE: Math.max(0, Math.round(1000 - i * 2 - (i > 300 ? (i - 300) * 5 : 0))),
      ANOMALY_FLAG: avionicsFlag
    };

    data.push(row);
  }

  return data;
}

// Convert FlightDataRow to EngineData array
export function getEngineDataFromRow(row: FlightDataRow): EngineData[] {
  const historicalMeans = [520, 525, 518, 522];

  return [
    {
      id: 1,
      egt: row.EGT_1,
      n1: row.N1_1,
      n2: row.N2_1,
      ff: row.FF_1,
      vib: row.VIB_1,
      oilTemp: row.OIL_TEMP_1,
      oilPress: row.OIL_PRESS_1,
      historicalMeanEGT: historicalMeans[0],
      rulScore: row.RUL_SCORE
    },
    {
      id: 2,
      egt: row.EGT_2,
      n1: row.N1_2,
      n2: row.N2_2,
      ff: row.FF_2,
      vib: row.VIB_2,
      oilTemp: row.OIL_TEMP_2,
      oilPress: row.OIL_PRESS_2,
      historicalMeanEGT: historicalMeans[1],
      rulScore: row.RUL_SCORE
    },
    {
      id: 3,
      egt: row.EGT_3,
      n1: row.N1_3,
      n2: row.N2_3,
      ff: row.FF_3,
      vib: row.VIB_3,
      oilTemp: row.OIL_TEMP_3,
      oilPress: row.OIL_PRESS_3,
      historicalMeanEGT: historicalMeans[2],
      rulScore: row.RUL_SCORE
    },
    {
      id: 4,
      egt: row.EGT_4,
      n1: row.N1_4,
      n2: row.N2_4,
      ff: row.FF_4,
      vib: row.VIB_4,
      oilTemp: row.OIL_TEMP_4,
      oilPress: row.OIL_PRESS_4,
      historicalMeanEGT: historicalMeans[3],
      rulScore: row.RUL_SCORE
    }
  ];
}

// Generate XAI insight based on current data
export function generateXAIInsight(row: FlightDataRow): XAIInsight {
  const vib2 = row.VIB_2;
  const vrtg = row.VRTG;

  if (vib2 > 2.0 && vrtg < 1.2) {
    return {
      failureProbability: Math.min(95, Math.round((vib2 - 2.0) * 30 + 40)),
      primaryCause: 'INTERNAL MECHANICAL WEAR',
      explanation: `AeroGuard detected a vibration spike (VIB_2: ${vib2.toFixed(2)}) but cross-referenced it with VRTG (G-load: ${vrtg.toFixed(2)}). Since VRTG is stable at ~1.0G, this is categorized as INTERNAL MECHANICAL WEAR rather than external turbulence.`,
      featureImportance: [
        { feature: 'Vibration', responsibility: 75 },
        { feature: 'Oil Temp', responsibility: 15 },
        { feature: 'Pressure', responsibility: 10 }
      ]
    };
  }

  return {
    failureProbability: 5,
    primaryCause: 'NORMAL OPERATION',
    explanation: 'All parameters within nominal ranges. No anomalies detected.',
    featureImportance: [
      { feature: 'Vibration', responsibility: 30 },
      { feature: 'Oil Temp', responsibility: 35 },
      { feature: 'Pressure', responsibility: 35 }
    ]
  };
}

// Get maintenance action recommendation
export function getMaintenanceAction(): MaintenanceAction {
  return {
    ammReference: 'AMM 72-31-00',
    partNumber: 'CS-901',
    partName: 'Compressor Seal',
    stockCount: 4,
    estimatedCost: 12500
  };
}

// Mock inventory data
export const mockInventory = {
  'CS-901': { name: 'Compressor Seal', count: 4, location: 'Hangar B-12' },
  'BR-205': { name: 'Bearing Assembly', count: 2, location: 'Hangar A-03' },
  'TC-447': { name: 'Turbine Blade Set', count: 0, location: 'External Supplier' }
};

// Calculate Data Quality Index
export function calculateDQI(row: FlightDataRow): number {
  let missingCount = 0;
  let totalFields = 0;

  Object.values(row).forEach(value => {
    totalFields++;
    if (value === null || value === undefined || value === 0) {
      missingCount++;
    }
  });

  return Math.round((1 - missingCount / totalFields) * 100);
}

// Convert FlightDataRow to AircraftSystemHealth for comprehensive error detection
export function getSystemHealthFromRow(row: FlightDataRow): import('@/types/aviation').AircraftSystemHealth {
  return {
    engines: getEngineDataFromRow(row),
    vrtg: row.VRTG,
    latg: row.LATG,
    hydy: row.HYDY,
    hydg: row.HYDG,
    flaps: (row.ELEV_1 + row.ELEV_2) / 2, // Average elevator position as flap proxy
    ailerons: (row.AIL_1 + row.AIL_2) / 2, // Average aileron position
    rudder: row.RUDD,
    elevator: (row.ELEV_1 + row.ELEV_2) / 2,
    // Determine avionics status based on data quality and anomaly flag
    gpsStatus: row.ANOMALY_FLAG > 0 ? 'warning' : 'nominal',
    throttleStatus: 'nominal', // Could be enhanced with additional logic
    activeErrors: [] // Will be populated by detectSystemErrors
  };
}
