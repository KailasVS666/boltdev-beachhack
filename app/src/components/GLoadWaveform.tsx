import { motion } from 'framer-motion';
import { Line, XAxis, YAxis, ReferenceLine, ResponsiveContainer, Area, ComposedChart } from 'recharts';
import type { FlightDataRow } from '@/types/aviation';
import { Activity, AlertTriangle } from 'lucide-react';

interface GLoadWaveformProps {
    historicalData: FlightDataRow[];
    currentVRTG: number;
}

export function GLoadWaveform({ historicalData, currentVRTG }: GLoadWaveformProps) {
    // Prepare chart data from last 60 samples
    const chartData = historicalData.slice(-60).map((row, index) => ({
        time: index,
        vrtg: row.VRTG,
        latg: row.LATG || 0,
    }));

    // Determine alert state
    const isTurbulence = currentVRTG > 1.3;
    const isSevere = currentVRTG > 1.5;

    // Calculate min/max for display
    const minG = Math.min(...chartData.map(d => d.vrtg), 0.8);
    const maxG = Math.max(...chartData.map(d => d.vrtg), 1.6);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="aviation-card p-6 h-full"
        >
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
                    <span className={`w-1 h-4 rounded-full ${isSevere ? 'bg-[#FF1744]' : isTurbulence ? 'bg-[#FFB300]' : 'bg-[#00E5FF]'}`} />
                    STRUCTURAL G-LOAD
                </h2>
                <div className="flex items-center gap-2">
                    {isTurbulence && (
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="flex items-center gap-1 px-2 py-1 rounded bg-[#FFB300]/20 border border-[#FFB300]/50"
                        >
                            <AlertTriangle className="w-3 h-3 text-[#FFB300]" />
                            <span className="text-[11px] font-mono text-[#FFB300]">TURBULENCE</span>
                        </motion.div>
                    )}
                    <Activity className="w-4 h-4 text-[#64748B]" />
                </div>
            </div>

            {/* Current G-Load Display */}
            <div className="flex items-center justify-between mb-4">
                <div>
                    <span className="text-xs text-[#64748B] uppercase tracking-wider">Vertical G</span>
                    <div className="flex items-end gap-1">
                        <span className={`text-3xl font-mono font-bold ${isSevere ? 'text-[#FF1744]' :
                            isTurbulence ? 'text-[#FFB300]' :
                                'text-[#00E5FF]'
                            }`}>
                            {currentVRTG.toFixed(2)}
                        </span>
                        <span className="text-sm text-[#64748B] mb-1">G</span>
                    </div>
                </div>

                {/* Threshold Legend */}
                <div className="flex flex-col gap-1 items-end">
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-px bg-[#FF1744]" />
                        <span className="text-[11px] text-[#64748B]">1.5G Severe</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-px bg-[#FFB300]" style={{ borderStyle: 'dashed', borderWidth: '1px 0 0 0' }} />
                        <span className="text-[11px] text-[#64748B]">1.3G Moderate</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-6 h-px bg-[#00E5FF]" />
                        <span className="text-[11px] text-[#64748B]">1.0G Nominal</span>
                    </div>
                </div>
            </div>

            {/* Waveform Chart */}
            <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                        <defs>
                            <linearGradient id="gloadGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#00E5FF" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#00E5FF" stopOpacity={0} />
                            </linearGradient>
                        </defs>

                        <XAxis
                            dataKey="time"
                            hide
                        />
                        <YAxis
                            domain={[minG - 0.1, maxG + 0.1]}
                            tick={{ fill: '#64748B', fontSize: 11 }}
                            tickFormatter={(val) => `${val.toFixed(1)}`}
                            width={35}
                        />

                        {/* Reference lines */}
                        <ReferenceLine y={1.0} stroke="#00E5FF" strokeWidth={1} strokeOpacity={0.5} />
                        <ReferenceLine y={1.3} stroke="#FFB300" strokeWidth={1} strokeDasharray="4 4" strokeOpacity={0.5} />
                        <ReferenceLine y={1.5} stroke="#FF1744" strokeWidth={1} strokeOpacity={0.5} />

                        {/* Area fill */}
                        <Area
                            type="monotone"
                            dataKey="vrtg"
                            stroke="none"
                            fill="url(#gloadGradient)"
                        />

                        {/* Main line */}
                        <Line
                            type="monotone"
                            dataKey="vrtg"
                            stroke={isSevere ? '#FF1744' : isTurbulence ? '#FFB300' : '#00E5FF'}
                            strokeWidth={2}
                            dot={false}
                            animationDuration={300}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* Time axis label */}
            <div className="flex justify-between mt-1">
                <span className="text-[11px] text-[#64748B]">-30s</span>
                <span className="text-[11px] text-[#64748B]">NOW</span>
            </div>

            {/* Causal Filter Note */}
            {isTurbulence && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 p-2 bg-[#FFB300]/10 border border-[#FFB300]/30 rounded text-xs text-[#FFB300]"
                >
                    <strong>CAUSAL FILTER:</strong> Elevated G-load detected. Engine vibration readings will be cross-referenced with turbulence data.
                </motion.div>
            )}
        </motion.div>
    );
}
