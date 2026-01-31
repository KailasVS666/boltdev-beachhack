import { motion } from 'framer-motion';
import type { EngineData } from '@/types/aviation';
import { AlertTriangle, TrendingUp } from 'lucide-react';

interface PropulsionPillarProps {
  engineData: EngineData[];
}

// Thermometer Gauge Component
function ThermometerGauge({ engine }: { engine: EngineData }) {
  const maxEGT = 700; // Maximum EGT for scale
  const fillPercentage = (engine.egt / maxEGT) * 100;
  const meanPercentage = (engine.historicalMeanEGT / maxEGT) * 100;

  // Determine state based on EGT deviation
  const egtDeviation = (engine.egt - engine.historicalMeanEGT) / engine.historicalMeanEGT;
  const isAlert = egtDeviation > 0.05;
  const isCrisis = engine.vib > 3.0;

  const getFillColor = () => {
    if (isCrisis) return 'from-[#FF1744] to-[#D50000]';
    if (isAlert) return 'from-[#FFB300] to-[#FF8F00]';
    return 'from-[#00E5FF] to-[#00B8D4]';
  };

  const getStateClass = () => {
    if (isCrisis) return 'state-crisis';
    if (isAlert) return 'state-alert';
    return 'state-nominal';
  };

  // Check for efficiency leak (FF increases but N1 stays same)
  const efficiencyLeak = engine.ff > 2600 && engine.n1 < 88;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: engine.id * 0.1, duration: 0.5 }}
      className="flex flex-col items-center gap-3"
    >
      {/* Engine Label */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-[#64748B]">ENG</span>
        <span className={`text-sm font-bold font-mono ${getStateClass()}`}>
          {engine.id}
        </span>
        {isAlert && <AlertTriangle className="w-3 h-3 text-[#FFB300]" />}
      </div>

      {/* Thermometer Bar */}
      <div className="relative w-12 h-48 thermometer-bar rounded-sm">
        {/* Historical Mean Line */}
        <div
          className="absolute left-0 right-0 border-t border-dashed border-white/30 z-10"
          style={{ bottom: `${meanPercentage}%` }}
        />
        <span
          className="absolute right-1 text-[8px] text-white/50 font-mono"
          style={{ bottom: `${meanPercentage}%`, transform: 'translateY(-50%)' }}
        >
          μ
        </span>

        {/* Fill */}
        <motion.div
          className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t ${getFillColor()} rounded-sm`}
          initial={{ height: 0 }}
          animate={{ height: `${fillPercentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        />

        {/* Scale Markers */}
        {[0, 25, 50, 75, 100].map(pct => (
          <div
            key={pct}
            className="absolute left-0 right-0 border-t border-[#1E293B]"
            style={{ bottom: `${pct}%` }}
          />
        ))}

        {/* EGT Value */}
        <div className="absolute -right-14 top-1/2 -translate-y-1/2">
          <span className={`text-base font-mono font-bold ${getStateClass()}`}>
            {Math.round(engine.egt)}
          </span>
          <span className="text-xs text-[#64748B] ml-1">°C</span>
        </div>
      </div>

      {/* Secondary Data */}
      <div className="flex flex-col gap-1 mt-2">
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-[#64748B]">N1</span>
          <span className="text-xs font-mono text-white">{engine.n1}%</span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-[#64748B]">FF</span>
          <span className={`text-xs font-mono ${efficiencyLeak ? 'text-[#FFB300]' : 'text-white'}`}>
            {Math.round(engine.ff)}
            {efficiencyLeak && <TrendingUp className="w-3 h-3 inline ml-1" />}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-[#64748B]">VIB</span>
          <span className={`text-xs font-mono ${engine.vib > 2.0 ? 'text-[#FF1744]' : 'text-white'}`}>
            {engine.vib.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Efficiency Leak Alert */}
      {efficiencyLeak && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="px-2 py-1 bg-[#FFB300]/20 border border-[#FFB300]/50 rounded text-[11px] text-[#FFB300] font-mono"
        >
          EFF LEAK
        </motion.div>
      )}

      {/* RUL Score */}
      <div className="mt-2 pt-2 border-t border-[#1E293B] w-full">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-[#64748B]">RUL</span>
          <span className={`text-xs font-mono ${engine.rulScore < 500 ? 'text-[#FF1744]' : 'text-[#00E5FF]'}`}>
            {engine.rulScore}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

export function PropulsionPillar({ engineData }: PropulsionPillarProps) {
  return (
    <div className="aviation-card p-6 h-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
          <span className="w-1 h-4 bg-[#00E5FF] rounded-full" />
          PROPULSION PILLAR
        </h2>
        <span className="text-xs text-[#64748B] font-mono">4 ENGINES ACTIVE</span>
      </div>

      <div className="flex justify-around items-end h-[calc(100%-40px)]">
        {engineData.map(engine => (
          <ThermometerGauge key={engine.id} engine={engine} />
        ))}
      </div>
    </div>
  );
}
