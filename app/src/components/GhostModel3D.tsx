import { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { Eye, EyeOff, Maximize2 } from 'lucide-react';

import type { EngineData } from '@/types/aviation';
import { Button } from '@/components/ui/button';
import { Boeing787Model } from './Boeing787Model';
import { SensorNode, type SensorStatus } from './SensorNode';
import { EngineMarker } from './EngineMarker';
import { detectSystemErrors } from '@/utils/errorDetection';
import { ErrorBanner } from '@/components/ErrorBanner';

interface GhostModel3DProps {
  engineData: EngineData[];
  systemHealth: import('@/types/aviation').AircraftSystemHealth;
  selectedEngine: number | null;
  onEngineSelect: (engineId: number) => void;
}

// Sensor coordinate mapping for Boeing 787 systems (adjusted for model scale)
const SENSOR_POSITIONS = {
  pitot: [0, 2, 15] as [number, number, number],        // Air Data Unit (Pitot) - nose
  radar: [0, 1.5, 16] as [number, number, number],      // Navigation Radar - front
  aoa: [3, 2, 12] as [number, number, number],          // Stability Vanes (AoA) - wing
  avionics: [0, -1, 8] as [number, number, number],     // Avionics Center (EE Bay) - belly
};

// Engine positions on wings (Boeing 787 has 2 engines)
const ENGINE_POSITIONS = {
  engine1: [-10, -0.8, 2] as [number, number, number],     // Left wing engine (on nacelle)
  engine2: [10, -0.8, 2] as [number, number, number],      // Right wing engine (on nacelle)
};

// Default camera position for recenter (Front-Right-High perspective)
const DEFAULT_CAMERA_POSITION = new THREE.Vector3(30, 30, 45);
const DEFAULT_CAMERA_TARGET = new THREE.Vector3(0, 0, 0);

// Map engine data to sensor status
function getSensorStatus(engineData: EngineData[], sensorType: string): SensorStatus {
  // For this demo, we'll use Engine 2 data as the primary diagnostic source
  const engine2 = engineData.find(e => e.id === 2);
  if (!engine2) return 'nominal';

  switch (sensorType) {
    case 'pitot':
      // Airspeed sensor - check for anomalies
      return engine2.vib > 2.5 ? 'warning' : 'nominal';

    case 'radar':
      // Navigation - nominal unless critical engine failure
      return engine2.vib > 3.0 ? 'critical' : 'nominal';

    case 'aoa':
      // Angle of Attack - check vibration and EGT
      if (engine2.vib > 3.0) return 'critical';
      if (engine2.vib > 2.0) return 'warning';
      return 'nominal';

    case 'avionics':
      // Avionics bay - check EGT deviation
      const egtDeviation = (engine2.egt - engine2.historicalMeanEGT) / engine2.historicalMeanEGT;
      if (egtDeviation > 0.1) return 'critical';
      if (egtDeviation > 0.05) return 'warning';
      return 'nominal';

    default:
      return 'nominal';
  }
}

// Scene Component
function Scene({
  engineData,
  selectedEngine,
  onEngineSelect,
  viewMode,
  recenterTrigger,
  systemHealth
}: GhostModel3DProps & {
  viewMode: 'skeletal' | 'solid';
  recenterTrigger: number;
  systemHealth: import('@/types/aviation').AircraftSystemHealth;
}) {
  const { camera } = useThree();
  const [selectedSensor, setSelectedSensor] = useState<string | null>(null);
  const controlsRef = useRef<any>(null);

  // Camera focus animation when sensor is selected
  useEffect(() => {
    if (selectedSensor && camera) {
      const sensorKey = selectedSensor as keyof typeof SENSOR_POSITIONS;
      const pos = SENSOR_POSITIONS[sensorKey];
      const targetPos = new THREE.Vector3(pos[0] + 8, pos[1] + 4, pos[2] + 8);

      const startPos = camera.position.clone();
      const duration = 1000;
      const startTime = Date.now();

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);

        camera.position.lerpVectors(startPos, targetPos, eased);
        camera.lookAt(pos[0], pos[1], pos[2]);

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      animate();
    }
  }, [selectedSensor, camera]);

  // Recenter camera animation
  useEffect(() => {
    if (recenterTrigger > 0 && camera && controlsRef.current) {
      const startPos = camera.position.clone();
      const startTarget = controlsRef.current.target.clone();
      const duration = 1000;
      const startTime = Date.now();

      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);

        camera.position.lerpVectors(startPos, DEFAULT_CAMERA_POSITION, eased);
        controlsRef.current.target.lerpVectors(startTarget, DEFAULT_CAMERA_TARGET, eased);
        controlsRef.current.update();

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      animate();
    }
  }, [recenterTrigger, camera]);

  const handleSensorClick = (sensorKey: string) => {
    setSelectedSensor(sensorKey === selectedSensor ? null : sensorKey);
  };

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.6} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#00E5FF" />
      <spotLight position={[0, 15, 0]} intensity={0.5} angle={0.3} penumbra={1} />

      {/* Boeing 787 GLB Model with Multi-System Error Detection */}
      <Boeing787Model viewMode={viewMode} systemHealth={systemHealth} />

      {/* Sensor Nodes */}
      <SensorNode
        systemName="Air Data Unit"
        status={getSensorStatus(engineData, 'pitot')}
        liveValue="Airspeed: 450kts"
        position={SENSOR_POSITIONS.pitot}
        onClick={() => handleSensorClick('pitot')}
        isSelected={selectedSensor === 'pitot'}
      />

      <SensorNode
        systemName="Navigation Radar"
        status={getSensorStatus(engineData, 'radar')}
        liveValue="Range: 120nm"
        position={SENSOR_POSITIONS.radar}
        onClick={() => handleSensorClick('radar')}
        isSelected={selectedSensor === 'radar'}
      />

      <SensorNode
        systemName="Angle of Attack"
        status={getSensorStatus(engineData, 'aoa')}
        liveValue="AoA: 4.2°"
        position={SENSOR_POSITIONS.aoa}
        onClick={() => handleSensorClick('aoa')}
        isSelected={selectedSensor === 'aoa'}
      />

      <SensorNode
        systemName="Avionics Center"
        status={getSensorStatus(engineData, 'avionics')}
        liveValue="Bus: 115V AC"
        position={SENSOR_POSITIONS.avionics}
        onClick={() => handleSensorClick('avionics')}
        isSelected={selectedSensor === 'avionics'}
      />

      {/* Engine Markers with Alert Visualization */}
      {engineData.map((engine) => {
        const engineKey = `engine${engine.id}` as keyof typeof ENGINE_POSITIONS;
        const position = ENGINE_POSITIONS[engineKey];
        if (!position) return null;

        return (
          <EngineMarker
            key={engine.id}
            engine={engine}
            position={position}
            onClick={() => onEngineSelect(engine.id)}
          />
        );
      })}

      <OrbitControls
        ref={controlsRef}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={15}
        maxDistance={300}
        target={[0, 0, 0]}
      />

      {/* Grid Helper */}
      <gridHelper args={[50, 50, '#1E293B', '#0B1120']} position={[0, -5, 0]} />
    </>
  );
}

export function GhostModel3D({ engineData, systemHealth, selectedEngine, onEngineSelect }: GhostModel3DProps) {
  const [viewMode, setViewMode] = useState<'skeletal' | 'solid'>('skeletal');
  const [recenterTrigger, setRecenterTrigger] = useState(0);

  // Detect active errors for the banner - use stable reference to prevent re-renders
  // Only update when error signature actually changes, not on every systemHealth update
  const activeErrors = useMemo(() => {
    const errors = detectSystemErrors(systemHealth);
    return errors;
  }, [
    // Only re-compute when critical values change, not the entire object
    systemHealth.engines.map(e => `${e.id}:${e.vib}:${e.egt}:${e.n1}`).join(','),
    systemHealth.hydy,
    systemHealth.hydg,
    systemHealth.vrtg,
    systemHealth.flaps,
    systemHealth.rudder,
    systemHealth.gpsStatus
  ]);

  const handleRecenter = () => {
    setRecenterTrigger(prev => prev + 1);
  };

  return (
    <div className="aviation-card h-full relative">
      <ErrorBanner errors={activeErrors} />

      <div className="absolute top-4 left-4 z-10">
        <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
          <span className="w-1 h-4 bg-[#00E5FF] rounded-full" />
          3D DIAGNOSTIC
        </h2>
        <p className="text-xs text-[#64748B] mt-1">Click sensors to inspect</p>
      </div>

      {/* View Mode Toggle and Recenter */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        <Button
          onClick={handleRecenter}
          variant="outline"
          size="sm"
          className="h-8 border-[#1E293B] text-[#64748B] hover:text-white hover:bg-[#1E293B]"
          title="Reset camera view"
        >
          <Maximize2 className="w-3 h-3 mr-1" />
          Recenter
        </Button>

        <Button
          onClick={() => setViewMode(prev => prev === 'skeletal' ? 'solid' : 'skeletal')}
          variant="outline"
          size="sm"
          className="h-8 border-[#1E293B] text-[#64748B] hover:text-white hover:bg-[#1E293B]"
        >
          {viewMode === 'skeletal' ? (
            <>
              <Eye className="w-3 h-3 mr-1" />
              Skeletal
            </>
          ) : (
            <>
              <EyeOff className="w-3 h-3 mr-1" />
              Solid
            </>
          )}
        </Button>
      </div>

      {/* Status Legend */}
      <div className="absolute top-14 right-4 z-10 flex flex-col gap-2 bg-[#0B1120]/80 backdrop-blur-sm border border-[#1E293B] rounded-lg p-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#00E5FF]" />
          <span className="text-xs text-[#64748B]">Nominal</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#FFB300]" />
          <span className="text-xs text-[#64748B]">Warning</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#FF1744]" />
          <span className="text-xs text-[#64748B]">Critical</span>
        </div>
      </div>

      <div className="canvas-3d h-full">
        <Canvas camera={{ position: [20, 20, 20], fov: 50 }}>
          <Scene
            engineData={engineData}
            selectedEngine={selectedEngine}
            onEngineSelect={onEngineSelect}
            viewMode={viewMode}
            recenterTrigger={recenterTrigger}
            systemHealth={systemHealth}
          />
        </Canvas>
      </div>
    </div>
  );
}
