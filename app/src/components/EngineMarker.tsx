import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import type { EngineData } from '@/types/aviation';

interface EngineMarkerProps {
    engine: EngineData;
    position: [number, number, number];
    onClick?: () => void;
}

export function EngineMarker({ engine, position, onClick }: EngineMarkerProps) {
    const ringRef = useRef<THREE.Mesh>(null);
    const glowRef = useRef<THREE.Mesh>(null);

    // Determine color based on engine status
    const getColor = () => {
        if (engine.vib > 3.0) return '#FF1744'; // Critical - red
        if (engine.vib > 2.0) return '#FFB300'; // Warning - yellow
        return '#00E5FF'; // Nominal - cyan
    };

    const color = getColor();
    const isCritical = engine.vib > 3.0;
    const isWarning = engine.vib > 2.0 && engine.vib <= 3.0;

    // Animation based on status
    useFrame((state) => {
        if (!ringRef.current || !glowRef.current) return;

        const time = state.clock.elapsedTime;

        if (isCritical) {
            // Rapid pulsing for critical
            const pulse = Math.sin(time * 8) * 0.5 + 0.5;
            ringRef.current.scale.setScalar(1 + pulse * 0.5);
            glowRef.current.scale.setScalar(2 + pulse * 0.8);
            if (glowRef.current.material instanceof THREE.MeshBasicMaterial) {
                glowRef.current.material.opacity = 0.6 + pulse * 0.3;
            }
        } else if (isWarning) {
            // Medium pulsing for warning
            const pulse = Math.sin(time * 4) * 0.5 + 0.5;
            ringRef.current.scale.setScalar(1 + pulse * 0.3);
            glowRef.current.scale.setScalar(1.8 + pulse * 0.4);
            if (glowRef.current.material instanceof THREE.MeshBasicMaterial) {
                glowRef.current.material.opacity = 0.4 + pulse * 0.2;
            }
        } else {
            // Subtle breathing for nominal
            const pulse = Math.sin(time * 2) * 0.5 + 0.5;
            ringRef.current.scale.setScalar(1 + pulse * 0.1);
            glowRef.current.scale.setScalar(1.5 + pulse * 0.2);
            if (glowRef.current.material instanceof THREE.MeshBasicMaterial) {
                glowRef.current.material.opacity = 0.2 + pulse * 0.1;
            }
        }
    });

    return (
        <group position={position}>
            {/* Engine Ring Marker */}
            <mesh ref={ringRef} onClick={onClick} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[0.5, 0.08, 16, 32]} />
                <meshBasicMaterial color={color} transparent opacity={0.8} />
            </mesh>

            {/* Glow Effect */}
            <mesh ref={glowRef} rotation={[Math.PI / 2, 0, 0]}>
                <torusGeometry args={[0.5, 0.12, 16, 32]} />
                <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={0.3}
                    side={THREE.BackSide}
                />
            </mesh>

            {/* Center Dot */}
            <mesh position={[0, 0, 0]}>
                <sphereGeometry args={[0.15, 16, 16]} />
                <meshBasicMaterial color={color} />
            </mesh>
        </group>
    );
}
