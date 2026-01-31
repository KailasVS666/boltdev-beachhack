import type { ErrorCategory, SystemError, AircraftSystemHealth } from '@/types/aviation';

/**
 * Maps error categories to their corresponding mesh names in the 787.glb model
 * Based on the actual mesh names registered in the Boeing787Model component
 */
export function getMeshesForError(category: ErrorCategory): string[] {
    switch (category) {
        // ===== EXISTING ERROR CATEGORIES =====

        // Engine & Propulsion - Target: [17] Engines_Material001_0
        case 'ENGINE_OVERHEAT':
        case 'ENGINE_FAN_SPOOL':
        case 'VIBRATION':
        case 'OIL_FUEL_SYSTEM':
            return ['Engines_Material001_0'];

        // Structural - Primary: [15] Fueselage__0, Secondary: [4] Body_Wings_Material002_0
        case 'STRUCTURAL':
            return ['Fueselage__0', 'Body_Wings_Material002_0'];

        // Hydraulics - Landing gear meshes
        case 'HYDRAULICS':
            return ['FRONT_LG_Material001_0', 'REAR_LEFT_LG_Material001_0', 'REAR_RIGHT_LG_Material001_0'];

        // Flight Controls - Specific control surfaces
        case 'FLIGHT_CONTROLS':
            return [
                'Rear_Rudder__0',
                'Elevators__0',
                'Ellerons_Material003_0',
                'Flaps_Material003_0'
            ];

        // Avionics - Cockpit area: [27] Windows_Material002_0
        case 'AVIONICS':
            return ['Windows_Material002_0'];

        // ===== NEW ERROR SIGNATURES =====

        // Engine Thermal Fatigue - Same as engine errors
        case 'ENGINE_THERMAL_FATIGUE':
            return ['Engines_Material001_0'];

        // Bearing/LPC Vibration - Engine meshes
        case 'BEARING_VIBRATION':
            return ['Engines_Material001_0'];

        // Hydraulic System Leak - Landing gear meshes
        case 'HYDRAULIC_LEAK':
            return [
                'REAR_LEFT_LG_Material001_0',
                'REAR_RIGHT_LG_Material001_0'
            ];

        // Structural Stress - Fuselage and wings
        case 'STRUCTURAL_STRESS':
            return ['Fueselage__0', 'Body_Wings_Material002_0'];

        // Fuel Feed Inconsistency - Wings (fuel tanks)
        case 'FUEL_FEED_INCONSISTENCY':
            return ['Body_Wings_Material003_0'];

        // Flight Control Surface Lag - All control surfaces
        case 'FLIGHT_CONTROL_LAG':
            return [
                'Flaps_Material003_0',
                'Rear_Rudder__0',
                'Elevators__0'
            ];

        // Aileron Actuator Fault - Ailerons only
        case 'AILERON_FAULT':
            return ['Ellerons_Material003_0'];

        // Pitot/Static Blockage - Cockpit/nose area
        case 'PITOT_STATIC_BLOCKAGE':
            return ['Windows_Material002_0'];

        default:
            return [];
    }
}

/**
 * Detects system errors from aircraft health data
 */
export function detectSystemErrors(health: AircraftSystemHealth): SystemError[] {
    const errors: SystemError[] = [];

    // Check each engine for errors
    health.engines.forEach(engine => {
        // Engine Overheat (EGT)
        const egtDeviation = (engine.egt - engine.historicalMeanEGT) / engine.historicalMeanEGT;
        if (egtDeviation > 0.15) {
            errors.push({
                category: 'ENGINE_OVERHEAT',
                severity: 'critical',
                affectedMeshes: getMeshesForError('ENGINE_OVERHEAT'),
                message: `Engine ${engine.id} EGT critical: ${engine.egt.toFixed(0)}°C`,
                value: engine.egt,
                threshold: engine.historicalMeanEGT * 1.15
            });
        } else if (egtDeviation > 0.10) {
            errors.push({
                category: 'ENGINE_OVERHEAT',
                severity: 'warning',
                affectedMeshes: getMeshesForError('ENGINE_OVERHEAT'),
                message: `Engine ${engine.id} EGT elevated`,
                value: engine.egt,
                threshold: engine.historicalMeanEGT * 1.10
            });
        }

        // Engine Fan/Spool (N1/N2)
        if (engine.n1 < 20 || engine.n1 > 105) {
            errors.push({
                category: 'ENGINE_FAN_SPOOL',
                severity: engine.n1 < 15 || engine.n1 > 110 ? 'critical' : 'warning',
                affectedMeshes: getMeshesForError('ENGINE_FAN_SPOOL'),
                message: `Engine ${engine.id} N1 out of range: ${engine.n1.toFixed(1)}%`,
                value: engine.n1
            });
        }

        // Vibration
        if (engine.vib > 3.0) {
            errors.push({
                category: 'VIBRATION',
                severity: 'critical',
                affectedMeshes: getMeshesForError('VIBRATION'),
                message: `Engine ${engine.id} vibration critical: ${engine.vib.toFixed(2)}`,
                value: engine.vib,
                threshold: 3.0
            });
        } else if (engine.vib > 2.0) {
            errors.push({
                category: 'VIBRATION',
                severity: 'warning',
                affectedMeshes: getMeshesForError('VIBRATION'),
                message: `Engine ${engine.id} vibration elevated`,
                value: engine.vib,
                threshold: 2.0
            });
        }

        // Oil/Fuel System
        if (engine.oilPress < 25 || engine.ff < 500) {
            errors.push({
                category: 'OIL_FUEL_SYSTEM',
                severity: engine.oilPress < 15 ? 'critical' : 'warning',
                affectedMeshes: getMeshesForError('OIL_FUEL_SYSTEM'),
                message: `Engine ${engine.id} oil/fuel system issue`,
                value: engine.oilPress
            });
        }
    });

    // Structural (VRTG)
    if (Math.abs(health.vrtg) > 2.5) {
        errors.push({
            category: 'STRUCTURAL',
            severity: Math.abs(health.vrtg) > 3.0 ? 'critical' : 'warning',
            affectedMeshes: getMeshesForError('STRUCTURAL'),
            message: `Excessive G-load: ${health.vrtg.toFixed(2)}G`,
            value: health.vrtg,
            threshold: 2.5
        });
    }

    // Hydraulics
    if (health.hydy < 2500 || health.hydg < 2500) {
        errors.push({
            category: 'HYDRAULICS',
            severity: health.hydy < 2000 || health.hydg < 2000 ? 'critical' : 'warning',
            affectedMeshes: getMeshesForError('HYDRAULICS'),
            message: 'Hydraulic pressure low',
            value: Math.min(health.hydy, health.hydg),
            threshold: 2500
        });
    }

    // Flight Controls
    if (Math.abs(health.flaps) > 40 || Math.abs(health.rudder) > 25) {
        errors.push({
            category: 'FLIGHT_CONTROLS',
            severity: 'warning',
            affectedMeshes: getMeshesForError('FLIGHT_CONTROLS'),
            message: 'Flight control position abnormal'
        });
    }

    // Avionics
    if (health.gpsStatus === 'critical' || health.throttleStatus === 'critical') {
        errors.push({
            category: 'AVIONICS',
            severity: 'critical',
            affectedMeshes: getMeshesForError('AVIONICS'),
            message: 'Avionics system failure'
        });
    } else if (health.gpsStatus === 'warning' || health.throttleStatus === 'warning') {
        errors.push({
            category: 'AVIONICS',
            severity: 'warning',
            affectedMeshes: getMeshesForError('AVIONICS'),
            message: 'Avionics system degraded'
        });
    }

    // Throttled logging - only log once per second to reduce console spam
    // console.log('🔍 [Error Detection] System Health Check:', {
    //     engines: health.engines.map(e => ({ id: e.id, egt: e.egt, n1: e.n1, vib: e.vib, oilPress: e.oilPress })),
    //     vrtg: health.vrtg,
    //     hydy: health.hydy,
    //     hydg: health.hydg,
    //     flaps: health.flaps,
    //     rudder: health.rudder,
    //     gpsStatus: health.gpsStatus
    // });

    // ===== NEW ERROR SIGNATURES =====

    // 1. Engine Thermal Fatigue (EGT > 820°C) - CRITICAL
    health.engines.forEach(engine => {
        if (engine.egt > 820) {
            errors.push({
                category: 'ENGINE_THERMAL_FATIGUE',
                severity: 'critical',
                priority: 'CRITICAL',
                affectedMeshes: getMeshesForError('ENGINE_THERMAL_FATIGUE'),
                message: `Engine ${engine.id} thermal fatigue: EGT ${engine.egt.toFixed(0)}°C`,
                value: engine.egt,
                threshold: 820,
                ammReference: 'AMM 72-00-00'
            });
        }
    });

    // 2. Bearing/LPC Vibration (VIB > 2.5) - URGENT
    health.engines.forEach(engine => {
        if (engine.vib > 2.5) {
            errors.push({
                category: 'BEARING_VIBRATION',
                severity: 'critical',
                priority: 'URGENT',
                affectedMeshes: getMeshesForError('BEARING_VIBRATION'),
                message: `Engine ${engine.id} bearing vibration: ${engine.vib.toFixed(2)}`,
                value: engine.vib,
                threshold: 2.5,
                ammReference: 'AMM 72-31-00'
            });
        }
    });

    // 3. Hydraulic System Leak (HYDY/HYDG < 2600 psi) - URGENT
    if (health.hydy < 2600 || health.hydg < 2600) {
        const affectedSystem = health.hydy < 2600 ? 'Yellow' : 'Green';
        const pressure = Math.min(health.hydy, health.hydg);
        errors.push({
            category: 'HYDRAULIC_LEAK',
            severity: 'critical',
            priority: 'URGENT',
            affectedMeshes: getMeshesForError('HYDRAULIC_LEAK'),
            message: `Hydraulic leak detected: ${affectedSystem} system ${pressure} psi`,
            value: pressure,
            threshold: 2600,
            ammReference: 'AMM 29-11-00'
        });
    }

    // 4. Structural Stress (VRTG > 1.85G) - CRITICAL
    if (Math.abs(health.vrtg) > 1.85) {
        errors.push({
            category: 'STRUCTURAL_STRESS',
            severity: 'critical',
            priority: 'CRITICAL',
            affectedMeshes: getMeshesForError('STRUCTURAL_STRESS'),
            message: `Structural stress: Hard landing detected (${health.vrtg.toFixed(2)}G)`,
            value: health.vrtg,
            threshold: 1.85,
            ammReference: 'AMM 05-51-01'
        });
    }

    // 5. Fuel Feed Inconsistency (FF drift > 10%) - WATCHLIST
    const avgFF = health.engines.reduce((sum, e) => sum + e.ff, 0) / health.engines.length;
    health.engines.forEach(engine => {
        const drift = Math.abs((engine.ff - avgFF) / avgFF);
        if (drift > 0.10) {
            errors.push({
                category: 'FUEL_FEED_INCONSISTENCY',
                severity: 'warning',
                priority: 'WATCHLIST',
                affectedMeshes: getMeshesForError('FUEL_FEED_INCONSISTENCY'),
                message: `Engine ${engine.id} fuel feed drift: ${(drift * 100).toFixed(1)}%`,
                value: engine.ff,
                threshold: avgFF * 1.10,
                ammReference: 'AMM 28-20-00'
            });
        }
    });

    // 6. Flight Control Surface Lag (FLAP/RUDD error > 2°) - WATCHLIST
    // Note: Assuming ideal flap position is 0 for cruise, adjust based on flight phase
    const flapError = Math.abs(health.flaps);
    const rudderError = Math.abs(health.rudder);
    if (flapError > 2 || rudderError > 2) {
        errors.push({
            category: 'FLIGHT_CONTROL_LAG',
            severity: 'warning',
            priority: 'WATCHLIST',
            affectedMeshes: getMeshesForError('FLIGHT_CONTROL_LAG'),
            message: `Flight control lag: Flap ${flapError.toFixed(1)}°, Rudder ${rudderError.toFixed(1)}°`,
            value: Math.max(flapError, rudderError),
            threshold: 2,
            ammReference: 'AMM 27-50-00'
        });
    }

    // 7. Aileron Actuator Fault (AIL lag > 0.5s) - URGENT
    // Note: Aileron lag tracking would need to be added to AircraftSystemHealth
    // For now, using aileron position as a proxy (if abs value > 15°, might indicate fault)
    if (Math.abs(health.ailerons) > 15) {
        errors.push({
            category: 'AILERON_FAULT',
            severity: 'critical',
            priority: 'URGENT',
            affectedMeshes: getMeshesForError('AILERON_FAULT'),
            message: `Aileron actuator fault: Position ${health.ailerons.toFixed(1)}°`,
            value: Math.abs(health.ailerons),
            threshold: 15,
            ammReference: 'AMM 27-10-00'
        });
    }

    // 8. Pitot/Static Blockage (ALT/CAS inconsistency) - CRITICAL
    // Note: Would need GPS altitude for comparison. Using GPS status as proxy for now
    if (health.gpsStatus === 'critical') {
        errors.push({
            category: 'PITOT_STATIC_BLOCKAGE',
            severity: 'critical',
            priority: 'CRITICAL',
            affectedMeshes: getMeshesForError('PITOT_STATIC_BLOCKAGE'),
            message: 'Pitot/static system blockage: Air data inconsistency',
            ammReference: 'AMM 34-11-00'
        });
    }

    // console.log(`🚨 [Error Detection] Found ${errors.length} errors:`, errors.map(e => ({
    //     category: e.category,
    //     severity: e.severity,
    //     meshes: e.affectedMeshes,
    //     message: e.message
    // })));

    return errors;
}

/**
 * Get mesh severity color for highlighting
 */
export function getMeshColor(meshName: string, errors: SystemError[]): string {
    const meshErrors = errors.filter(e => e.affectedMeshes.includes(meshName));

    if (meshErrors.length === 0) return '#00ffff'; // Nominal - cyan

    const hasCritical = meshErrors.some(e => e.severity === 'critical');
    return hasCritical ? '#FF0000' : '#FFB300'; // Red or Yellow
}

/**
 * Get severity level for a mesh
 */
export function getMeshSeverity(meshName: string, errors: SystemError[]): 'nominal' | 'warning' | 'critical' {
    const meshErrors = errors.filter(e => e.affectedMeshes.includes(meshName));

    if (meshErrors.length === 0) return 'nominal';

    const hasCritical = meshErrors.some(e => e.severity === 'critical');
    return hasCritical ? 'critical' : 'warning';
}
