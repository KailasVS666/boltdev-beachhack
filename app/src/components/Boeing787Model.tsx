import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';
import type { AircraftSystemHealth } from '@/types/aviation';
import { detectSystemErrors } from '@/utils/errorDetection';

interface Boeing787ModelProps {
    viewMode: 'skeletal' | 'solid';
    systemHealth: AircraftSystemHealth;
}

// Categorize mesh by position to determine which engine it belongs to (for wireframes)
function categorizeMesh(mesh: THREE.Mesh): 'leftEngine' | 'rightEngine' | 'fuselage' | 'wings' | 'tail' | 'other' {
    const worldPos = new THREE.Vector3();
    mesh.getWorldPosition(worldPos);

    const x = worldPos.x;
    const z = worldPos.z;

    // Left engine area (negative X, around engine position)
    if (x < -5 && Math.abs(z - 2) < 4) return 'leftEngine';

    // Right engine area (positive X, around engine position)
    if (x > 5 && Math.abs(z - 2) < 4) return 'rightEngine';

    // Fuselage (center area)
    if (Math.abs(x) < 3) return 'fuselage';

    // Wings (wide X, mid Z)
    if (Math.abs(x) > 3 && Math.abs(z) < 8) return 'wings';

    // Tail (far back Z)
    if (z < -5) return 'tail';

    return 'other';
}

export function Boeing787Model({ viewMode, systemHealth }: Boeing787ModelProps) {
    const { scene } = useGLTF('/787.glb');
    const leftEngineWireframeRef = useRef<THREE.Group>(null);
    const rightEngineWireframeRef = useRef<THREE.Group>(null);
    const otherWireframeRef = useRef<THREE.Group>(null);

    // Store references to all critical meshes for error-based highlighting
    const meshRefsMap = useRef<Map<string, THREE.Mesh>>(new Map());

    // Cache materials to avoid creating new ones every frame (performance optimization)
    const glowingMaterialsCache = useRef<Map<string, THREE.MeshStandardMaterial>>(new Map());
    const previousErrorsRef = useRef<string>('');

    // Detect active system errors
    const activeErrors = useMemo(() => detectSystemErrors(systemHealth), [systemHealth]);

    // X-Ray Glass Hull Material
    const xrayMaterial = useMemo(() => {
        return new THREE.MeshPhysicalMaterial({
            transmission: 1.0,
            roughness: 0.05,
            thickness: 2.0,
            opacity: 0.15,
            transparent: true,
            color: new THREE.Color('#0B1120'),
            metalness: 0.1,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1,
            envMapIntensity: 0.5,
        });
    }, []);

    // Solid Material
    const solidMaterial = useMemo(() => {
        return new THREE.MeshStandardMaterial({
            color: new THREE.Color('#1E293B'),
            metalness: 0.8,
            roughness: 0.3,
        });
    }, []);

    // Clone scene and categorize meshes
    const modelGroup = useMemo(() => {
        const clonedScene = scene.clone();
        const leftEngineWireframes: THREE.Mesh[] = [];
        const rightEngineWireframes: THREE.Mesh[] = [];
        const otherWireframes: THREE.Mesh[] = [];
        const processedGeometries = new Set<THREE.BufferGeometry>();
        const meshMap = new Map<string, THREE.Mesh>();

        console.log('🔍 [Boeing787Model] Initializing aircraft meshes...');

        clonedScene.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                // Store reference to all named meshes for error-based highlighting
                if (child.name) {
                    meshMap.set(child.name, child);
                    console.log(`  Registered mesh: "${child.name}"`);
                }

                // Hide passenger windows
                const isWindow = child.name.toLowerCase().includes('window') ||
                    child.name.toLowerCase().includes('glass') ||
                    (child.geometry.attributes.position.count < 100 &&
                        child.scale.x < 0.5 && child.scale.y < 0.5);

                if (isWindow) {
                    child.visible = false;
                    return;
                }

                // Apply initial material to all meshes
                child.material = viewMode === 'skeletal' ? xrayMaterial : solidMaterial;
                child.visible = true;

                // Create wireframes categorized by location
                if (viewMode === 'skeletal' && !processedGeometries.has(child.geometry)) {
                    processedGeometries.add(child.geometry);

                    const wireframeMaterial = new THREE.MeshBasicMaterial({
                        color: new THREE.Color('#00ffff'),
                        wireframe: true,
                        transparent: true,
                        opacity: 0.6,
                    });

                    const wireframeMesh = new THREE.Mesh(child.geometry, wireframeMaterial);
                    wireframeMesh.position.copy(child.position);
                    wireframeMesh.rotation.copy(child.rotation);
                    wireframeMesh.scale.copy(child.scale);
                    wireframeMesh.matrixWorld.copy(child.matrixWorld);

                    // Store mesh name for error-based coloring
                    (wireframeMesh as any).originalMeshName = child.name;

                    // Check if this is an engine mesh by NAME first
                    const isEngineMesh = child.name === 'Engines_Material.001_0' || child.name.toLowerCase().includes('engine');

                    if (isEngineMesh) {
                        // Engine meshes - determine left vs right by spatial position
                        const worldPos = new THREE.Vector3();
                        child.getWorldPosition(worldPos);

                        if (worldPos.x < 0) {
                            leftEngineWireframes.push(wireframeMesh);
                            console.log(`  ⬅️ Engine wireframe to LEFT (x=${worldPos.x.toFixed(2)})`);
                        } else {
                            rightEngineWireframes.push(wireframeMesh);
                            console.log(`  ➡️ Engine wireframe to RIGHT (x=${worldPos.x.toFixed(2)})`);
                        }
                    } else {
                        // Non-engine meshes - use spatial categorization
                        const category = categorizeMesh(child);
                        if (category === 'leftEngine') {
                            leftEngineWireframes.push(wireframeMesh);
                        } else if (category === 'rightEngine') {
                            rightEngineWireframes.push(wireframeMesh);
                        } else {
                            otherWireframes.push(wireframeMesh);
                        }
                    }
                }
            }
        });

        console.log(`🎯 [Boeing787Model] Registered ${meshMap.size} meshes`);
        console.log(`🎯 [Boeing787Model] Left engine wireframes: ${leftEngineWireframes.length}`);
        console.log(`🎯 [Boeing787Model] Right engine wireframes: ${rightEngineWireframes.length}`);

        return {
            model: clonedScene,
            leftEngineWireframes,
            rightEngineWireframes,
            otherWireframes,
            meshMap // Return the map so we can access it
        };
    }, [scene, viewMode, xrayMaterial, solidMaterial]);

    // Populate mesh references AFTER model is created
    useEffect(() => {
        if (modelGroup.meshMap) {
            meshRefsMap.current = modelGroup.meshMap;
            console.log('✅ [Boeing787Model] Mesh map populated with', meshRefsMap.current.size, 'meshes');
            console.log('📋 [Boeing787Model] Available meshes:', Array.from(meshRefsMap.current.keys()));
        }
    }, [modelGroup]);

    // Animate wireframes and meshes based on errors
    useFrame((state) => {
        const time = state.clock.elapsedTime;

        // Throttle logging to once per second
        const shouldLog = Math.floor(time) !== Math.floor(time - 0.016);

        // Update wireframe colors based on active errors
        if (viewMode === 'skeletal') {
            const leftVib = systemHealth.engines.find(e => e.id === 1)?.vib || 0;
            const rightVib = systemHealth.engines.find(e => e.id === 2)?.vib || 0;

            const updateWireframeGroup = (group: THREE.Group | null, vib: number, engineName: string) => {
                if (!group) return;
                const isCritical = vib > 3.0;
                const isWarning = vib > 2.0;
                const speed = isCritical ? 10 : (isWarning ? 5 : 2);
                const color = isCritical ? '#FF0000' : (isWarning ? '#FFB300' : '#00ffff');
                const intensity = 0.6 + Math.sin(time * speed) * 0.3;

                let meshCount = 0;
                group.traverse((child) => {
                    if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshBasicMaterial) {
                        child.material.color.set(color);
                        child.material.opacity = intensity;
                        child.material.needsUpdate = true;
                        meshCount++;
                    }
                });

                if (isCritical && meshCount > 0) {
                    console.log(`🔴 [${engineName}] Wireframe RED - ${meshCount} meshes updated`);
                }
            };

            updateWireframeGroup(leftEngineWireframeRef.current, leftVib, 'Left Engine');
            updateWireframeGroup(rightEngineWireframeRef.current, rightVib, 'Right Engine');

            if (otherWireframeRef.current) {
                const intensity = 0.4 + Math.sin(time * 2) * 0.2;
                otherWireframeRef.current.traverse((child) => {
                    if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshBasicMaterial) {
                        child.material.opacity = intensity;
                    }
                });
            }
        }

        // Update mesh materials based on active errors
        if (activeErrors.length > 0 && meshRefsMap.current.size > 0) {
            // Only log when errors change (reduce console spam)
            const errorSignature = activeErrors.map(e => `${e.category}:${e.severity}`).join(',');
            const hasErrorsChanged = errorSignature !== previousErrorsRef.current;

            if (hasErrorsChanged) {
                console.log(`🎨 [Boeing787Model] Processing ${activeErrors.length} errors`);
                previousErrorsRef.current = errorSignature;
            }

            activeErrors.forEach(error => {
                if (hasErrorsChanged) {
                    console.log(`  📍 Error: ${error.category} (${error.severity}) → Meshes:`, error.affectedMeshes);
                }

                error.affectedMeshes.forEach(meshName => {
                    const mesh = meshRefsMap.current.get(meshName);
                    if (!mesh) {
                        if (hasErrorsChanged) {
                            console.warn(`    ⚠️ Mesh "${meshName}" NOT FOUND in registry`);
                        }
                        return;
                    }

                    // Determine color and speed based on priority (new) or severity (fallback)
                    let color: string;
                    let speed: number;

                    if (error.priority) {
                        // New priority-based system
                        switch (error.priority) {
                            case 'CRITICAL':
                                color = '#FF0000';  // Red
                                speed = 10;
                                break;
                            case 'URGENT':
                                color = '#FF6B00';  // Orange
                                speed = 7;
                                break;
                            case 'WATCHLIST':
                                color = '#FFB300';  // Amber
                                speed = 4;
                                break;
                            default:
                                color = '#FF0000';
                                speed = 10;
                        }
                    } else {
                        // Fallback to old severity system
                        color = error.severity === 'critical' ? '#FF0000' : '#FFB300';
                        speed = error.severity === 'critical' ? 10 : 5;
                    }
                    const intensity = 0.5 + Math.sin(time * speed) * 0.5;

                    // Cache key for this mesh-priority/severity combination
                    const materialKey = `${meshName}-${error.priority || error.severity}`;

                    // Get or create cached material
                    let glowMaterial = glowingMaterialsCache.current.get(materialKey);
                    if (!glowMaterial) {
                        glowMaterial = new THREE.MeshStandardMaterial({
                            color: new THREE.Color(color),
                            emissive: new THREE.Color(color),
                            emissiveIntensity: 2 + intensity * 8,
                            metalness: 0.5,
                            roughness: 0.2,
                            side: THREE.DoubleSide,
                            transparent: false,
                            opacity: 1.0,
                        });
                        glowingMaterialsCache.current.set(materialKey, glowMaterial);

                        if (hasErrorsChanged) {
                            const priorityLabel = error.priority || error.severity.toUpperCase();
                            const ammRef = error.ammReference ? ` → ${error.ammReference}` : '';
                            console.log(`    ✅ Highlighting "${meshName}" with ${color} (${priorityLabel})${ammRef}`);
                        }
                    }

                    // Only update intensity (much cheaper than creating new material)
                    glowMaterial.emissiveIntensity = 2 + intensity * 8;
                    glowMaterial.needsUpdate = false; // Intensity doesn't need full update

                    mesh.material = glowMaterial;
                    mesh.visible = true;
                });
            });
        } else if (previousErrorsRef.current !== '') {
            // Errors cleared
            previousErrorsRef.current = '';
        }
    });

    return (
        <group rotation={[0, Math.PI, 0]} scale={0.015} position={[0, 0, 0]}>
            <primitive object={modelGroup.model} />

            {viewMode === 'skeletal' && (
                <>
                    <group ref={leftEngineWireframeRef}>
                        {modelGroup.leftEngineWireframes.map((wireframe, index) => (
                            <primitive key={`left-${index}`} object={wireframe} />
                        ))}
                    </group>
                    <group ref={rightEngineWireframeRef}>
                        {modelGroup.rightEngineWireframes.map((wireframe, index) => (
                            <primitive key={`right-${index}`} object={wireframe} />
                        ))}
                    </group>
                    <group ref={otherWireframeRef}>
                        {modelGroup.otherWireframes.map((wireframe, index) => (
                            <primitive key={`other-${index}`} object={wireframe} />
                        ))}
                    </group>
                </>
            )}
        </group>
    );
}

useGLTF.preload('/787.glb');
