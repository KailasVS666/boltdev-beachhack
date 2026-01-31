import { motion } from 'framer-motion';
import type { FlightDataRow } from '@/types/aviation';
import { Clock, Plane, Thermometer, Shield } from 'lucide-react';

interface VitalsStripProps {
  currentRow: FlightDataRow | null;
  dqi: number;
  systemStatus: 'nominal' | 'alert' | 'crisis';
}

// Progress Ring Component
function ProgressRing({ value, size = 40 }: { value: number; size?: number }) {
  const strokeWidth = 3;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;
  
  const getColor = () => {
    if (value >= 95) return '#00E5FF';
    if (value >= 80) return '#FFB300';
    return '#FF1744';
  };
  
  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#1E293B"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={getColor()}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-[10px] font-mono" style={{ color: getColor() }}>
          {value}
        </span>
      </div>
    </div>
  );
}

export function VitalsStrip({ currentRow, dqi, systemStatus }: VitalsStripProps) {
  if (!currentRow) return null;
  
  const formatTime = () => {
    const { GMT_HOUR, GMT_MINUTE, GMT_SEC } = currentRow;
    return `${String(GMT_HOUR).padStart(2, '0')}:${String(GMT_MINUTE).padStart(2, '0')}:${String(GMT_SEC).padStart(2, '0')} UTC`;
  };
  
  const getStatusColor = () => {
    switch (systemStatus) {
      case 'nominal': return 'text-[#00E5FF]';
      case 'alert': return 'text-[#FFB300]';
      case 'crisis': return 'text-[#FF1744]';
    }
  };
  
  const getStatusGlow = () => {
    switch (systemStatus) {
      case 'nominal': return 'glow-cyan';
      case 'alert': return 'glow-amber';
      case 'crisis': return 'glow-red';
    }
  };
  
  return (
    <motion.header
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="h-[60px] bg-[#050810]/95 border-b border-[#1E293B] flex items-center px-6 sticky top-0 z-50 backdrop-blur-md"
    >
      {/* Logo / Brand */}
      <div className="flex items-center gap-3 mr-8">
        <div className="w-8 h-8 rounded bg-gradient-to-br from-[#00E5FF] to-[#00B8D4] flex items-center justify-center">
          <Plane className="w-5 h-5 text-[#050810]" />
        </div>
        <span className="text-sm font-bold tracking-wider text-white">
          AERO<span className="text-[#00E5FF]">GUARD</span>
        </span>
      </div>
      
      {/* Widget A - Time */}
      <div className="flex items-center gap-2 px-4 border-r border-[#1E293B]">
        <Clock className="w-4 h-4 text-[#64748B]" />
        <span className={`font-mono text-sm ${getStatusColor()} ${getStatusGlow()}`}>
          {formatTime()}
        </span>
      </div>
      
      {/* Widget B - Environment */}
      <div className="flex items-center gap-4 px-4 border-r border-[#1E293B]">
        <div className="flex items-center gap-2">
          <Plane className="w-4 h-4 text-[#64748B]" />
          <span className="font-mono text-sm text-white">
            {(currentRow.ALT / 1000).toFixed(1)}<span className="text-[#64748B] text-xs">k ft</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Thermometer className="w-4 h-4 text-[#64748B]" />
          <span className="font-mono text-sm text-white">
            {currentRow.SAT}°<span className="text-[#64748B] text-xs">C</span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[#64748B] text-xs">GS</span>
          <span className="font-mono text-sm text-white">
            {Math.round(currentRow.GS)}
          </span>
        </div>
      </div>
      
      {/* Widget C - Data Quality Index */}
      <div className="flex items-center gap-3 px-4">
        <ProgressRing value={dqi} size={36} />
        <div className="flex flex-col">
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">DQI</span>
          {dqi === 100 ? (
            <div className="flex items-center gap-1">
              <Shield className="w-3 h-3 text-[#00E5FF]" />
              <span className="text-[10px] text-[#00E5FF] font-medium">CERTIFIED</span>
            </div>
          ) : (
            <span className="text-[10px] text-[#FFB300]">DEGRADED</span>
          )}
        </div>
      </div>
      
      {/* System Status Indicator */}
      <div className="ml-auto flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${
          systemStatus === 'nominal' ? 'bg-[#00E5FF]' : 
          systemStatus === 'alert' ? 'bg-[#FFB300]' : 'bg-[#FF1744]'
        } animate-pulse`} />
        <span className={`text-xs font-mono uppercase tracking-wider ${getStatusColor()}`}>
          {systemStatus}
        </span>
      </div>
    </motion.header>
  );
}
