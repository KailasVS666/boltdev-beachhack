import { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import * as THREE from 'three';
import { AlertTriangle } from 'lucide-react';

export type SensorStatus = 'nominal' | 'warning' | 'critical';

interface SensorNodeProps {
    systemName: string;
    status: SensorStatus;
    liveValue: string;
    position: [number, number, number];
    onClick?: () => void;
    isSelected?: boolean;
}

export function SensorNode({
    systemName,
    status,
    liveValue,
    position,
    onClick,
    isSelected = false
}: SensorNodeProps) {
    const meshRef = useRef<THREE.Mesh>(null);
    const [hovered, setHovered] = useState(false);

    // Color mapping based on status
    const getColor = () => {
        switch (status) {
            case 'critical': return '#FF1744';
            case 'warning': return '#FFB300';
            case 'nominal': return '#00E5FF';
            default: return '#00E5FF';
        }
    };

    const color = getColor();

    // Animation based on status
    useFrame((state) => {
        if (!meshRef.current) return;

        const time = state.clock.elapsedTime;

        switch (status) {
            case 'critical':
                // Rapid flashing for critical
                const flashIntensity = Math.sin(time * 10) > 0 ? 1.5 : 0.5;
                meshRef.current.scale.setScalar(0.3 * flashIntensity);
                break;

            case 'warning':
                // Pulsing for warning
                const pulseScale = 0.3 + Math.sin(time * 3) * 0.1;
                meshRef.current.scale.setScalar(pulseScale);
                break;

            case 'nominal':
                // Static with subtle breathing
                const breatheScale = 0.25 + Math.sin(time * 0.5) * 0.02;
                meshRef.current.scale.setScalar(breatheScale);
                break;
        }
    });

    return (
        <group position={position}>
            {/* Sensor Sphere */}
            <mesh
                ref={meshRef}
                onClick={onClick}
                onPointerOver={() => setHovered(true)}
                onPointerOut={() => setHovered(false)}
            >
                <sphereGeometry args={[1, 16, 16]} />
                <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={0.8}
                />
            </mesh>

            {/* Glow Effect */}
            <mesh scale={isSelected ? 2.5 : 1.8}>
                <sphereGeometry args={[1, 16, 16]} />
                <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={status === 'critical' ? 0.4 : 0.2}
                    side={THREE.BackSide}
                />
            </mesh>

            {/* Critical Alert Icon */}
            {status === 'critical' && (
                <Html center distanceFactor={10}>
                    <div className="flex items-center justify-center w-8 h-8 bg-[#FF1744]/20 rounded-full border border-[#FF1744] animate-pulse">
                        <AlertTriangle className="w-4 h-4 text-[#FF1744]" />
                    </div>
                </Html>
            )}

            {/* Hover/Selected Info Panel */}
            {(hovered || isSelected) && (
                <Html
                    position={[0, 1.5, 0]}
                    center
                    distanceFactor={8}
                    style={{ pointerEvents: 'none' }}
                >
                    <div className="bg-[#0B1120]/95 backdrop-blur-sm border border-[#1E293B] rounded-lg px-3 py-2 min-w-[150px] shadow-xl">
                        <div className="text-xs font-bold text-white mb-1 uppercase tracking-wider">
                            {systemName}
                        </div>
                        <div className="text-[11px] text-[#64748B] font-mono">
                            {liveValue}
                        </div>
                        <div className="flex items-center gap-1 mt-2">
                            <div
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-[10px] text-[#64748B] uppercase">
                                {status}
                            </span>
                        </div>
                    </div>
                </Html>
            )}
        </group>
    );
}
