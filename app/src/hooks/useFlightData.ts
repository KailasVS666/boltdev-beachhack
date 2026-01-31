import { useState, useEffect, useCallback, useRef } from 'react';
import type { FlightDataRow, EngineData, SystemState, ViewMode, XAIInsight, MaintenanceAction } from '@/types/aviation';
import {
  generateMockFlightData,
  getEngineDataFromRow,
  getSystemHealthFromRow,
  generateXAIInsight,
  getMaintenanceAction,
  calculateDQI
} from '@/data/mockFlightData';

interface UseFlightDataReturn {
  // Current data
  currentRow: FlightDataRow | null;
  engineData: EngineData[];
  systemHealth: import('@/types/aviation').AircraftSystemHealth | null;
  dqi: number;

  // System state
  systemState: SystemState;

  // Views
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  selectedEngine: number | null;
  setSelectedEngine: (engine: number | null) => void;

  // XAI & Maintenance
  xaiInsight: XAIInsight | null;
  maintenanceAction: MaintenanceAction | null;

  // Playback control
  isPlaying: boolean;
  togglePlayback: () => void;
  currentRowIndex: number;
  totalRows: number;

  // Feedback
  submitFeedback: (accepted: boolean, reason?: string) => void;

  // Historical data for trends
  historicalData: FlightDataRow[];
}

export function useFlightData(): UseFlightDataReturn {
  // Generate mock data once
  const flightDataRef = useRef<FlightDataRow[]>(generateMockFlightData());
  const flightData = flightDataRef.current;

  // Current state
  const [currentRowIndex, setCurrentRowIndex] = useState(0);
  const [currentRow, setCurrentRow] = useState<FlightDataRow | null>(flightData[0]);
  const [engineData, setEngineData] = useState<EngineData[]>(getEngineDataFromRow(flightData[0]));
  const [dqi, setDqi] = useState(100);

  // System state
  const [systemState, setSystemState] = useState<SystemState>({
    status: 'nominal',
    message: 'All systems nominal'
  });

  // View mode
  const [viewMode, setViewMode] = useState<ViewMode>('dashboard');
  const [selectedEngine, setSelectedEngine] = useState<number | null>(null);

  // XAI & Maintenance
  const [xaiInsight, setXaiInsight] = useState<XAIInsight | null>(null);
  const [maintenanceAction, setMaintenanceAction] = useState<MaintenanceAction | null>(null);

  // Playback
  const [isPlaying, setIsPlaying] = useState(true);

  // Historical data buffer (last 50 rows for trend analysis)
  const [historicalData, setHistoricalData] = useState<FlightDataRow[]>([flightData[0]]);

  // Check system state based on current data
  const checkSystemState = useCallback((row: FlightDataRow): SystemState => {
    // Check for crisis conditions first
    if (row.VIB_2 > 3.0 && row.VRTG < 1.2) {
      return {
        status: 'crisis',
        message: 'CRITICAL: Engine 2 vibration threshold exceeded',
        affectedEngine: 2
      };
    }

    // Check for alert conditions
    if (row.VIB_2 > 2.0 && row.VRTG < 1.2) {
      return {
        status: 'alert',
        message: 'ALERT: Engine 2 showing elevated vibration',
        affectedEngine: 2
      };
    }

    // Check EGT deviations
    const egtDeviation = (row.EGT_2 - 525) / 525;
    if (egtDeviation > 0.05) {
      return {
        status: 'alert',
        message: `ALERT: Engine 2 EGT ${(egtDeviation * 100).toFixed(1)}% above mean`,
        affectedEngine: 2
      };
    }

    return {
      status: 'nominal',
      message: 'All systems nominal'
    };
  }, []);

  // Data heartbeat - read new row every 500ms
  useEffect(() => {
    if (!isPlaying) return;

    // --- INTEGRATION: Poll Requestly API ---
    const fetchBackendDiagnosis = async () => {
      try {
        const response = await fetch('http://localhost:8000/diagnosis');
        if (response.ok) {
          const report = await response.json();
          console.log("📡 Backend Diagnosis Received:", report);

          if (report.overall_status === 'SAFETY BREACH DETECTED') {
            setSystemState({
              status: 'crisis',
              message: `CRITICAL: ${report.diagnosis_details.root_cause_diagnosis}`,
              affectedEngine: 2 // Determined by parsing logical root cause
            });

            // Map backend diagnosis to XAI Insight
            setXaiInsight({
              failureProbability: 95, // High confidence on crisis
              primaryCause: report.diagnosis_details.root_cause_diagnosis,
              explanation: report.diagnosis_details.evidence_summary,
              featureImportance: [
                { feature: 'FLAP', responsibility: 75 },
                { feature: 'VIB', responsibility: 15 },
                { feature: 'EGT', responsibility: 10 }
              ]
            });

            setMaintenanceAction({
              ammReference: report.diagnosis_details.maintenance_action.amm_reference,
              partNumber: "ACT-FLAP-787-01", // Fallback / Simulated
              partName: "Flap Actuator Assembly",
              stockCount: 2,
              estimatedCost: 12500
            });

            // Only auto-switch if not already there
            if (viewMode === 'dashboard') {
              setViewMode('diagnostic');
            }
          }
        }
      } catch (e) {
        // Ignore fetch errors (Requestly not set up yet)
      }
    };

    // Poll backend every 2 seconds
    const backendInterval = setInterval(fetchBackendDiagnosis, 2000);

    const interval = setInterval(() => {
      setCurrentRowIndex(prev => {
        const nextIndex = (prev + 1) % flightData.length;
        const row = flightData[nextIndex];

        setCurrentRow(row);
        setEngineData(getEngineDataFromRow(row));
        setDqi(calculateDQI(row));

        // Update system state (Mock logic serves as fallback if API fails)
        const newState = checkSystemState(row);
        // Only override if not already set by Backend API
        setSystemState(prev => prev.status === 'crisis' ? prev : newState);

        // Update XAI insight if anomaly detected
        if (row.ANOMALY_FLAG === 1 || row.VIB_2 > 2.0) {
          // Keep existing mock logic as fallback
          if (!xaiInsight) setXaiInsight(generateXAIInsight(row));
          if (!maintenanceAction) setMaintenanceAction(getMaintenanceAction());
        }

        // Update historical data
        setHistoricalData(prev => {
          const newData = [...prev, row];
          if (newData.length > 50) {
            return newData.slice(newData.length - 50);
          }
          return newData;
        });

        // Auto-switch to diagnostic view on crisis onset
        setSystemState(prev => {
          if (newState.status === 'crisis' && prev.status !== 'crisis' && viewMode === 'dashboard') {
            setViewMode('diagnostic');
            setSelectedEngine(newState.affectedEngine || null);
          }
          return newState;
        });

        return nextIndex;
      });
    }, 500);

    return () => {
      clearInterval(interval);
      clearInterval(backendInterval);
    };
  }, [isPlaying, flightData, checkSystemState, viewMode]);

  // Toggle playback
  const togglePlayback = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  // Submit feedback
  const submitFeedback = useCallback((accepted: boolean, reason?: string) => {
    // In a real app, this would save to engineer_feedback.csv
    console.log('Feedback submitted:', { accepted, reason, timestamp: new Date().toISOString() });

    // Reset maintenance action after feedback
    if (accepted) {
      setMaintenanceAction(null);
      setViewMode('dashboard');
    }
  }, []);

  return {
    currentRow,
    engineData,
    systemHealth: currentRow ? getSystemHealthFromRow(currentRow) : null,
    dqi,
    systemState,
    viewMode,
    setViewMode,
    selectedEngine,
    setSelectedEngine,
    xaiInsight,
    maintenanceAction,
    isPlaying,
    togglePlayback,
    currentRowIndex,
    totalRows: flightData.length,
    submitFeedback,
    historicalData
  };
}
