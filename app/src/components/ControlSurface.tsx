import { motion } from 'framer-motion';

interface ControlSurfaceProps {
    roll: number;   // -180 to 180 degrees
    pitch: number;  // -30 to 30 degrees
    rudder: number; // -30 to 30 degrees
}

export function ControlSurface({ roll, pitch, rudder }: ControlSurfaceProps) {
    // Clamp values for safety
    const clampedRoll = Math.max(-30, Math.min(30, roll));
    const clampedPitch = Math.max(-15, Math.min(15, pitch));
    const clampedRudder = Math.max(-30, Math.min(30, rudder));

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="aviation-card p-6 h-full relative overflow-hidden"
        >
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
                    <span className="w-1 h-4 bg-[#00E5FF] rounded-full" />
                    CONTROL SURFACE
                </h2>
                <span className="text-xs text-[#64748B] font-mono">FCS ACTIVE</span>
            </div>

            {/* Aircraft Silhouette Container */}
            <div className="flex items-center justify-center h-[calc(100%-80px)]">
                <motion.div
                    className="relative"
                    style={{ width: 200, height: 180 }}
                    animate={{
                        rotateZ: clampedRoll,
                        rotateX: -clampedPitch * 0.3
                    }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                >
                    {/* Aircraft SVG */}
                    <svg viewBox="0 0 200 180" className="w-full h-full">
                        {/* Fuselage */}
                        <motion.path
                            d="M100 10 L110 60 L110 140 L105 170 L95 170 L90 140 L90 60 Z"
                            fill="#0B1120"
                            stroke="#00E5FF"
                            strokeWidth="1.5"
                            className="drop-shadow-lg"
                        />

                        {/* Cockpit */}
                        <ellipse cx="100" cy="25" rx="8" ry="12" fill="#1E293B" stroke="#00E5FF" strokeWidth="1" />

                        {/* Left Wing */}
                        <motion.path
                            d="M90 70 L10 90 L10 95 L90 85 Z"
                            fill="#0B1120"
                            stroke="#00E5FF"
                            strokeWidth="1.5"
                            animate={{
                                opacity: roll > 0 ? 0.6 : 1,
                            }}
                        />

                        {/* Right Wing */}
                        <motion.path
                            d="M110 70 L190 90 L190 95 L110 85 Z"
                            fill="#0B1120"
                            stroke="#00E5FF"
                            strokeWidth="1.5"
                            animate={{
                                opacity: roll < 0 ? 0.6 : 1,
                            }}
                        />

                        {/* Left Horizontal Stabilizer */}
                        <path
                            d="M92 140 L55 150 L55 153 L92 145 Z"
                            fill="#0B1120"
                            stroke="#64748B"
                            strokeWidth="1"
                        />

                        {/* Right Horizontal Stabilizer */}
                        <path
                            d="M108 140 L145 150 L145 153 L108 145 Z"
                            fill="#0B1120"
                            stroke="#64748B"
                            strokeWidth="1"
                        />

                        {/* Vertical Stabilizer / Rudder */}
                        <motion.path
                            d={`M${100 + clampedRudder * 0.3} 125 L100 150 L${102 + clampedRudder * 0.1} 170 L${98 + clampedRudder * 0.1} 170 L100 150 Z`}
                            fill="#0B1120"
                            stroke={Math.abs(clampedRudder) > 10 ? '#FFB300' : '#00E5FF'}
                            strokeWidth="1.5"
                            animate={{
                                d: `M${100 + clampedRudder * 0.3} 125 L100 150 L${102 + clampedRudder * 0.1} 170 L${98 + clampedRudder * 0.1} 170 L100 150 Z`
                            }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                        />

                        {/* Engine Indicators */}
                        <circle cx="35" cy="92" r="4" fill="#1E293B" stroke="#00E5FF" strokeWidth="1" />
                        <circle cx="165" cy="92" r="4" fill="#1E293B" stroke="#00E5FF" strokeWidth="1" />
                        <circle cx="55" cy="88" r="4" fill="#1E293B" stroke="#00E5FF" strokeWidth="1" />
                        <circle cx="145" cy="88" r="4" fill="#1E293B" stroke="#00E5FF" strokeWidth="1" />
                    </svg>

                    {/* Pitch indicator arrows */}
                    {Math.abs(clampedPitch) > 3 && (
                        <motion.div
                            className="absolute left-1/2 -translate-x-1/2"
                            style={{ top: clampedPitch > 0 ? -10 : 'auto', bottom: clampedPitch < 0 ? -10 : 'auto' }}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                        >
                            <span className={`text-sm font-mono font-bold ${clampedPitch > 0 ? 'text-[#00E5FF]' : 'text-[#FFB300]'}`}>
                                {clampedPitch > 0 ? '▲ NOSE UP' : '▼ NOSE DN'}
                            </span>
                        </motion.div>
                    )}
                </motion.div>
            </div>

            {/* Data Display */}
            <div className="absolute bottom-4 left-6 right-6 grid grid-cols-3 gap-4">
                <div className="text-center">
                    <span className="text-xs text-[#64748B] uppercase tracking-wider block">Roll</span>
                    <span className={`text-base font-mono font-bold ${Math.abs(roll) > 15 ? 'text-[#FFB300]' : 'text-white'}`}>
                        {roll > 0 ? '+' : ''}{roll.toFixed(1)}°
                    </span>
                </div>
                <div className="text-center">
                    <span className="text-xs text-[#64748B] uppercase tracking-wider block">Pitch</span>
                    <span className={`text-base font-mono font-bold ${Math.abs(pitch) > 10 ? 'text-[#FFB300]' : 'text-white'}`}>
                        {pitch > 0 ? '+' : ''}{pitch.toFixed(1)}°
                    </span>
                </div>
                <div className="text-center">
                    <span className="text-xs text-[#64748B] uppercase tracking-wider block">Rudder</span>
                    <span className={`text-base font-mono font-bold ${Math.abs(rudder) > 10 ? 'text-[#FFB300]' : 'text-white'}`}>
                        {rudder > 0 ? '+' : ''}{rudder.toFixed(1)}°
                    </span>
                </div>
            </div>
        </motion.div>
    );
}
