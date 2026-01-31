import { motion } from 'framer-motion';
import { PropulsionPillar } from '@/components/PropulsionPillar';
import { HydraulicGauges } from '@/components/HydraulicGauges';
import { ControlSurface } from '@/components/ControlSurface';
import { GLoadWaveform } from '@/components/GLoadWaveform';
import type { EngineData, FlightDataRow } from '@/types/aviation';
import { Activity, TrendingUp, Wind, Gauge } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

interface DashboardViewProps {
  engineData: EngineData[];
  historicalData: FlightDataRow[];
  currentRow: FlightDataRow | null;
}

// Mini Trend Chart Component
function MiniTrendChart({
  data,
  color
}: {
  data: { value: number; time: number }[];
  color: string;
}) {
  return (
    <div className="h-16 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={1.5}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Quick Stats Card
function QuickStatCard({
  title,
  value,
  unit,
  icon: Icon,
  trend,
  color = '#00E5FF'
}: {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ComponentType<{ className?: string; color?: string }>;
  trend?: 'up' | 'down' | 'stable';
  color?: string;
}) {
  return (
    <div className="aviation-card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-[#64748B] uppercase tracking-wider">{title}</span>
        <Icon className="w-4 h-4" color={color} />
      </div>
      <div className="flex items-end gap-1">
        <span className="text-xl font-mono font-bold text-white">{value}</span>
        {unit && <span className="text-xs text-[#64748B] mb-1">{unit}</span>}
      </div>
      {trend && (
        <div className="flex items-center gap-1 mt-1">
          <TrendingUp className={`w-3 h-3 ${trend === 'up' ? 'text-[#FF1744]' :
            trend === 'down' ? 'text-[#00E5FF]' :
              'text-[#64748B]'
            }`} />
          <span className="text-[11px] text-[#64748B]">{trend}</span>
        </div>
      )}
    </div>
  );
}

export function DashboardView({ engineData, historicalData, currentRow }: DashboardViewProps) {
  // Prepare trend data
  const vibTrendData = historicalData.map((row, i) => ({
    time: i,
    value: row.VIB_2
  }));

  const egtTrendData = historicalData.map((row, i) => ({
    time: i,
    value: row.EGT_2
  }));

  // Calculate average values
  const avgVibration = engineData.reduce((acc, eng) => acc + eng.vib, 0) / 4;
  const avgEGT = engineData.reduce((acc, eng) => acc + eng.egt, 0) / 4;
  const totalFF = engineData.reduce((acc, eng) => acc + eng.ff, 0);

  // Get current hydraulics and control data
  const hydy = currentRow?.HYDY ?? 2900;
  const hydg = currentRow?.HYDG ?? 2950;
  const roll = currentRow?.ROLL ?? 0;
  const pitch = currentRow?.PTCH ?? 0;
  const rudder = currentRow?.RUDD ?? 0;
  const currentVRTG = currentRow?.VRTG ?? 1.0;

  return (
    <motion.div
      initial={{ opacity: 0, x: -50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -50 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="min-h-full flex flex-col gap-4 p-4"
    >
      {/* Quick Stats Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
        <QuickStatCard
          title="Avg Vibration"
          value={avgVibration.toFixed(2)}
          unit="units"
          icon={Activity}
          trend={avgVibration > 1.5 ? 'up' : 'stable'}
          color={avgVibration > 2.0 ? '#FF1744' : avgVibration > 1.5 ? '#FFB300' : '#00E5FF'}
        />
        <QuickStatCard
          title="Avg EGT"
          value={Math.round(avgEGT)}
          unit="°C"
          icon={Gauge}
          color="#00E5FF"
        />
        <QuickStatCard
          title="Total Fuel Flow"
          value={Math.round(totalFF)}
          unit="pph"
          icon={Wind}
          color="#00E5FF"
        />
        <QuickStatCard
          title="G-Load"
          value={currentVRTG.toFixed(2)}
          unit="G"
          icon={Activity}
          color={currentVRTG > 1.3 ? '#FFB300' : '#00E5FF'}
        />
        <QuickStatCard
          title="HYD Yellow"
          value={hydy}
          unit="psi"
          icon={Gauge}
          color={hydy >= 2700 ? '#00E5FF' : '#FFB300'}
        />
        <QuickStatCard
          title="HYD Green"
          value={hydg}
          unit="psi"
          icon={Gauge}
          color={hydg >= 2700 ? '#00E5FF' : '#FFB300'}
        />
      </div>

      {/* MVP 2x2 Grid - The Four Pillars of Health */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 min-h-0">
        {/* Top-Left: Propulsion Matrix */}
        <div className="min-h-0">
          <PropulsionPillar engineData={engineData} />
        </div>

        {/* Top-Right: Hydraulic Sync */}
        <div className="min-h-0">
          <HydraulicGauges hydy={hydy} hydg={hydg} />
        </div>

        {/* Bottom-Left: Control Surface */}
        <div className="min-h-0">
          <ControlSurface roll={roll} pitch={pitch} rudder={rudder} />
        </div>

        {/* Bottom-Right: G-Load Waveform */}
        <div className="min-h-0">
          <GLoadWaveform historicalData={historicalData} currentVRTG={currentVRTG} />
        </div>
      </div>

      {/* Engine Trend Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Vibration Trend */}
        <div className="aviation-card p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-bold text-white tracking-wider flex items-center gap-2">
              <span className="w-1 h-3 bg-[#FF1744] rounded-full" />
              ENGINE 2 VIBRATION TREND
            </h3>
            <span className="text-xs text-[#64748B]">Last 50 samples</span>
          </div>
          <MiniTrendChart
            data={vibTrendData}
            color="#FF1744"
          />
          <div className="flex justify-between mt-2">
            <span className="text-[11px] text-[#64748B]">Threshold: 3.0</span>
            <span className={`text-sm font-mono font-bold ${engineData[1]?.vib > 3.0 ? 'text-[#FF1744]' :
              engineData[1]?.vib > 2.0 ? 'text-[#FFB300]' :
                'text-[#00E5FF]'
              }`}>
              Current: {engineData[1]?.vib.toFixed(2)}
            </span>
          </div>
        </div>

        {/* EGT Trend */}
        <div className="aviation-card p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-bold text-white tracking-wider flex items-center gap-2">
              <span className="w-1 h-3 bg-[#FFB300] rounded-full" />
              ENGINE 2 EGT TREND
            </h3>
            <span className="text-xs text-[#64748B]">Last 50 samples</span>
          </div>
          <MiniTrendChart
            data={egtTrendData}
            color="#FFB300"
          />
          <div className="flex justify-between mt-2">
            <span className="text-[11px] text-[#64748B]">Mean: 525°C</span>
            <span className={`text-sm font-mono font-bold ${engineData[1]?.egt > 550 ? 'text-[#FF1744]' :
              engineData[1]?.egt > 530 ? 'text-[#FFB300]' :
                'text-[#00E5FF]'
              }`}>
              Current: {engineData[1]?.egt}°C
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
