import { motion } from 'framer-motion';

interface HydraulicGaugesProps {
    hydy: number;  // Yellow System (0-3000 psi)
    hydg: number;  // Green System (0-3000 psi)
}

// Radial Gauge Component
function RadialGauge({
    value,
    label,
    color,
    maxValue = 3000
}: {
    value: number;
    label: string;
    color: string;
    maxValue?: number;
}) {
    const size = 120;
    const strokeWidth = 10;
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * Math.PI; // Half circle
    const percentage = (value / maxValue) * 100;
    const offset = circumference - (percentage / 100) * circumference;

    // Determine pressure zone color
    const getZoneColor = () => {
        if (value < 2500) return '#FF1744';  // Danger - low pressure
        if (value < 2700) return '#FFB300';  // Caution
        return color;                         // Nominal
    };

    const needleAngle = (percentage / 100) * 180 - 90; // -90 to 90 degrees

    return (
        <div className="flex flex-col items-center">
            <div className="relative" style={{ width: size, height: size / 2 + 20 }}>
                {/* Background Arc */}
                <svg
                    width={size}
                    height={size / 2 + 10}
                    viewBox={`0 0 ${size} ${size / 2 + 10}`}
                    className="absolute top-0 left-0"
                >
                    {/* Background track */}
                    <path
                        d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
                        fill="none"
                        stroke="#1E293B"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                    />

                    {/* Colored arc */}
                    <motion.path
                        d={`M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
                        fill="none"
                        stroke={getZoneColor()}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: offset }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                    />

                    {/* Zone markers */}
                    {[0, 2500, 2700, 3000].map((zone, i) => {
                        const zoneAngle = (zone / maxValue) * 180;
                        const x = size / 2 + Math.cos((zoneAngle - 180) * Math.PI / 180) * (radius + 15);
                        const y = size / 2 + Math.sin((zoneAngle - 180) * Math.PI / 180) * (radius + 15);
                        return (
                            <text
                                key={zone}
                                x={x}
                                y={y}
                                fill="#64748B"
                                fontSize="11"
                                textAnchor="middle"
                                dominantBaseline="middle"
                            >
                                {i === 0 ? '0' : i === 3 ? '3K' : ''}
                            </text>
                        );
                    })}
                </svg>

                {/* Needle */}
                <motion.div
                    className="absolute"
                    style={{
                        width: 4,
                        height: radius - 15,
                        background: `linear-gradient(to top, ${getZoneColor()}, transparent)`,
                        borderRadius: 2,
                        left: '50%',
                        bottom: 0,
                        transformOrigin: 'bottom center',
                    }}
                    initial={{ rotate: -90 }}
                    animate={{ rotate: needleAngle }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                />

                {/* Center dot */}
                <div
                    className="absolute w-3 h-3 rounded-full bg-[#0B1120] border-2"
                    style={{
                        borderColor: getZoneColor(),
                        left: '50%',
                        bottom: -2,
                        transform: 'translateX(-50%)'
                    }}
                />

                {/* Value display - Moved lower and added background blur/glow for readability */}
                <div
                    className="absolute text-center bg-[#050810]/40 backdrop-blur-sm px-2 py-0.5 rounded-full border border-white/5"
                    style={{ left: '50%', bottom: -30, transform: 'translateX(-50%)', minWidth: '80px' }}
                >
                    <span
                        className="text-xl font-mono font-bold tracking-tighter"
                        style={{
                            color: getZoneColor(),
                            textShadow: `0 0 10px ${getZoneColor()}44`
                        }}
                    >
                        {value}
                    </span>
                    <span className="text-[11px] text-[#64748B] ml-1 font-bold">PSI</span>
                </div>
            </div>

            {/* Label - Adjusted margin for new value position */}
            <div className="flex items-center gap-2 mt-10">
                <div
                    className="w-2.5 h-2.5 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.5)]"
                    style={{ backgroundColor: color, border: '1px solid rgba(255,255,255,0.1)' }}
                />
                <span className="text-xs font-mono font-bold text-[#94A3B8] uppercase tracking-widest">
                    {label}
                </span>
            </div>
        </div>
    );
}

export function HydraulicGauges({ hydy, hydg }: HydraulicGaugesProps) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="aviation-card p-6 h-full"
        >
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
                    <span className="w-1 h-4 bg-[#FFB300] rounded-full" />
                    HYDRAULIC SYNC
                </h2>
                <span className="text-xs text-[#64748B] font-mono">2 SYSTEMS</span>
            </div>

            <div className="flex justify-around items-center h-[calc(100%-40px)]">
                <RadialGauge
                    value={hydy}
                    label="Yellow Sys"
                    color="#FFB300"
                />
                <RadialGauge
                    value={hydg}
                    label="Green Sys"
                    color="#22C55E"
                />
            </div>

            {/* Status indicator */}
            <div className="absolute bottom-4 left-6 right-6 flex justify-between">
                <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded-full ${hydy >= 2700 ? 'bg-[#00E5FF]' : hydy >= 2500 ? 'bg-[#FFB300]' : 'bg-[#FF1744]'}`} />
                    <span className="text-[11px] text-[#64748B]">Y: {hydy >= 2700 ? 'NOM' : hydy >= 2500 ? 'CAU' : 'LOW'}</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className={`w-2 h-2 rounded-full ${hydg >= 2700 ? 'bg-[#00E5FF]' : hydg >= 2500 ? 'bg-[#FFB300]' : 'bg-[#FF1744]'}`} />
                    <span className="text-[11px] text-[#64748B]">G: {hydg >= 2700 ? 'NOM' : hydg >= 2500 ? 'CAU' : 'LOW'}</span>
                </div>
            </div>
        </motion.div>
    );
}
