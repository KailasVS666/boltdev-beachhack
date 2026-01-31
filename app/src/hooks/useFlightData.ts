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

    const interval = setInterval(() => {
      setCurrentRowIndex(prev => {
        const nextIndex = (prev + 1) % flightData.length;
        const row = flightData[nextIndex];

        setCurrentRow(row);
        setEngineData(getEngineDataFromRow(row));
        setDqi(calculateDQI(row));

        // Update system state
        const newState = checkSystemState(row);
        setSystemState(newState);

        // Update XAI insight if anomaly detected
        if (row.ANOMALY_FLAG === 1 || row.VIB_2 > 2.0) {
          setXaiInsight(generateXAIInsight(row));
          setMaintenanceAction(getMaintenanceAction());
        } else {
          setXaiInsight(null);
          setMaintenanceAction(null);
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

    return () => clearInterval(interval);
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
