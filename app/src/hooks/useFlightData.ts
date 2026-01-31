import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import type { FlightDataRow, EngineData, SystemState, ViewMode, XAIInsight, MaintenanceAction } from '@/types/aviation';
import {
  generateMockFlightData,
  getEngineDataFromRow,
  getSystemHealthFromRow,
  generateXAIInsight,
  getMaintenanceAction,
  calculateDQI
} from '@/data/mockFlightData';
import { detectSystemErrors } from '@/utils/errorDetection';
import type { SystemError } from '@/types/aviation';

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

  // Freeze lock to prevent XAI updates on minor value changes
  const xaiFrozenRef = useRef(false);

  // Error persistence tracking - prevent blinking
  const errorCountRef = useRef(0);
  const noErrorCountRef = useRef(0);
  const ERROR_THRESHOLD = 4; // Must see error 4 times (2 seconds) before showing

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

    // === ML MODE: Backend Integration ENABLED ===
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
              affectedEngine: 2
            });

            const causalChain = report.causal_chain || ['HYDY', 'FLAP', 'VIB'];
            const confidence = Math.round((report.confidence || 0.88) * 100);

            const featureImportance = causalChain.map((feature: string, index: number) => ({
              feature,
              responsibility: index === 0 ? 75 : index === 1 ? 15 : 10
            })).slice(0, 3);

            setXaiInsight({
              failureProbability: confidence,
              primaryCause: report.diagnosis_details.root_cause_diagnosis,
              explanation: report.diagnosis_details.evidence_summary,
              featureImportance
            });

            setMaintenanceAction({
              ammReference: report.diagnosis_details.maintenance_action.amm_reference,
              partNumber: report.diagnosis_details.maintenance_action.parts_required?.[0] || "ACT-FLAP-787-01",
              partName: report.diagnosis_details.maintenance_action.parts_required?.[0]?.includes('HYD')
                ? "Hydraulic Pump Assembly"
                : "Flap Actuator Assembly",
              stockCount: 2,
              estimatedCost: report.diagnosis_details.maintenance_action.estimated_hours
                ? Math.round(report.diagnosis_details.maintenance_action.estimated_hours * 3500)
                : 12500
            });

            xaiFrozenRef.current = true;

            if (viewMode === 'dashboard') {
              setViewMode('diagnostic');
            }
          }
          // NOTE: Don't clear on normal status - keep data stable for demo
        } else {
          console.warn("⚠️ Backend returned non-OK status:", response.status);
        }
      } catch (e) {
        console.error("❌ Backend API Error:", e);
        // Fallback handled by demo mode below
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

        // === DEMO MODE: Static Error Scenario - Show error IMMEDIATELY and keep visible ===
        if (nextIndex >= 5 && !xaiFrozenRef.current) {
          console.log("🎬 [DEMO MODE] Showing persistent error scenario - ALWAYS VISIBLE");

          setXaiInsight({
            failureProbability: 75,
            primaryCause: "Flight Control Lag Detected",
            explanation: `System detected FLIGHT_CONTROL_LAG. Aircraft control surfaces showing delayed response. Flaps registering 2.8° lag, Rudder showing 3.2° lag. Refer to AMM 27-50-00 for maintenance procedures. Recommended action: Inspect hydraulic actuators and control linkages.`,
            featureImportance: [
              { feature: 'FLIGHT', responsibility: 75 },
              { feature: 'CONTROL', responsibility: 15 },
              { feature: 'LAG', responsibility: 10 }
            ]
          });

          setMaintenanceAction({
            ammReference: "AMM 27-50-00",
            partNumber: "ACT-FLAP-787-01",
            partName: "Flight Control Actuator Assembly",
            stockCount: 2,
            estimatedCost: 12500
          });

          xaiFrozenRef.current = true;

          if (viewMode === 'dashboard') {
            setViewMode('diagnostic');
            setSelectedEngine(2);
          }
        }

        // === DEMO MODE: NEVER CLEAR - Keep data visible for entire demo ===
        // if (nextIndex >= 200 && xaiFrozenRef.current) {
        //   console.log("✅ [DEMO MODE] Clearing error scenario");
        //   setXaiInsight(null);
        //   setMaintenanceAction(null);
        //   xaiFrozenRef.current = false;
        // }

        // === DISABLED: Error detection for demo ===
        // Check for errors using the error detection system
        // const systemHealth = getSystemHealthFromRow(row);
        // const detectedErrors = detectSystemErrors(systemHealth);

        // Persistence-based error handling - prevent blinking
        // if (detectedErrors.length > 0) {
        //   errorCountRef.current++;
        //   noErrorCountRef.current = 0;

        //   // Only show error if it persists for threshold duration
        //   if (errorCountRef.current >= ERROR_THRESHOLD && !xaiFrozenRef.current) {
        //     const criticalError = detectedErrors.find((err: SystemError) => err.severity === 'critical') || detectedErrors[0];
        //     console.log("🔒 [Main Loop] Error PERSISTENT, LOCKING XAI data:", criticalError);

        //     setXaiInsight({
        //       failureProbability: criticalError.severity === 'critical' ? 85 : 65,
        //       primaryCause: criticalError.message,
        //       explanation: `System detected ${criticalError.category}. ${criticalError.ammReference ? `Refer to ${criticalError.ammReference} for maintenance procedures.` : 'Monitoring for additional symptoms.'}`,
        //       featureImportance: [
        //         { feature: criticalError.category.split('_')[0], responsibility: 75 },
        //         { feature: criticalError.category.split('_')[1] || 'SYS', responsibility: 15 },
        //         { feature: 'ENV', responsibility: 10 }
        //       ]
        //     });

        //     setMaintenanceAction({
        //       ammReference: criticalError.ammReference || "AMM-PENDING",
        //       partNumber: "PART-TBD",
        //       partName: criticalError.category.replace(/_/g, ' '),
        //       stockCount: 0,
        //       estimatedCost: 0
        //     });

        //     // LOCK the XAI - no more updates until cleared
        //     xaiFrozenRef.current = true;

        //     // Auto-switch to diagnostic view when error is first detected
        //     if (viewMode === 'dashboard') {
        //       setViewMode('diagnostic');
        //       setSelectedEngine(2);
        //     }
        //   }
        // } else {
        //   noErrorCountRef.current++;
        //   errorCountRef.current = 0;

        //   // Only clear if no errors for threshold duration
        //   if (noErrorCountRef.current >= ERROR_THRESHOLD && xaiFrozenRef.current) {
        //     console.log("✅ [Main Loop] No errors PERSISTENT, UNLOCKING XAI");
        //     setXaiInsight(null);
        //     setMaintenanceAction(null);
        //     xaiFrozenRef.current = false;
        //   }
        // }

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
    }, 1500); // DEMO MODE: Slower updates (1.5 seconds) for better readability

    return () => {
      clearInterval(interval);
      clearInterval(backendInterval); // ML MODE: Backend polling enabled
    };
  }, [isPlaying, flightData, viewMode]);

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

  // Memoize systemHealth to prevent 3D model rebuilds when values don't change
  const systemHealth = useMemo(() => {
    if (!currentRow) return null;
    return getSystemHealthFromRow(currentRow);
  }, [
    // Only recreate if actual values change
    currentRow?.VIB_1,
    currentRow?.VIB_2,
    currentRow?.VIB_3,
    currentRow?.VIB_4,
    currentRow?.EGT_1,
    currentRow?.EGT_2,
    currentRow?.EGT_3,
    currentRow?.EGT_4,
    currentRow?.N1_1,
    currentRow?.N1_2,
    currentRow?.N1_3,
    currentRow?.N1_4,
    currentRow?.VRTG,
    currentRow?.HYDY,
    currentRow?.HYDG,
  ]);

  return {
    currentRow,
    engineData,
    systemHealth,
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
