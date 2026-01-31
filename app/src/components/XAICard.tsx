import { motion } from 'framer-motion';
import type { XAIInsight } from '@/types/aviation';
import { Brain, AlertCircle, BarChart3, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';

interface XAICardProps {
  insight: XAIInsight | null;
}

export function XAICard({ insight }: XAICardProps) {
  if (!insight) {
    return (
      <div className="aviation-card p-6 h-full flex flex-col items-center justify-center">
        <Brain className="w-12 h-12 text-[#1E293B] mb-4" />
        <p className="text-sm text-[#64748B]">No anomalies detected</p>
        <p className="text-[10px] text-[#475569] mt-2">AI monitoring active...</p>
      </div>
    );
  }
  
  const isAnomaly = insight.failureProbability > 20;
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="aviation-card p-6 h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
          <span className="w-1 h-4 bg-[#00E5FF] rounded-full" />
          XAI CAUSAL LOGIC
        </h2>
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-[#00E5FF]" />
          <span className="text-[10px] text-[#64748B]">AI BRAIN</span>
        </div>
      </div>
      
      {/* Failure Probability */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Failure Probability</span>
          <span className={`text-2xl font-mono font-bold ${
            insight.failureProbability > 50 ? 'text-[#FF1744]' :
            insight.failureProbability > 20 ? 'text-[#FFB300]' :
            'text-[#00E5FF]'
          }`}>
            {insight.failureProbability}%
          </span>
        </div>
        <div className="h-2 bg-[#1E293B] rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${insight.failureProbability}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className={`h-full rounded-full ${
              insight.failureProbability > 50 ? 'bg-[#FF1744]' :
              insight.failureProbability > 20 ? 'bg-[#FFB300]' :
              'bg-[#00E5FF]'
            }`}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[9px] text-[#64748B]">0 cycles</span>
          <span className="text-[9px] text-[#64748B]">1000 cycles</span>
        </div>
      </div>
      
      {/* Primary Cause */}
      <div className="mb-6 p-4 bg-[#0B1120] rounded border border-[#1E293B]">
        <div className="flex items-center gap-2 mb-2">
          <AlertCircle className={`w-4 h-4 ${isAnomaly ? 'text-[#FF1744]' : 'text-[#00E5FF]'}`} />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Primary Cause</span>
        </div>
        <p className={`text-sm font-mono font-bold ${isAnomaly ? 'text-[#FF1744]' : 'text-[#00E5FF]'}`}>
          {insight.primaryCause}
        </p>
      </div>
      
      {/* Explanation */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="w-4 h-4 text-[#64748B]" />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Analysis</span>
        </div>
        <p className="text-xs text-[#94A3B8] leading-relaxed">
          {insight.explanation}
        </p>
      </div>
      
      {/* Feature Importance Chart */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-[#64748B]" />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Feature Importance</span>
        </div>
        
        <div className="h-32">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={insight.featureImportance} layout="vertical">
              <XAxis type="number" domain={[0, 100]} hide />
              <YAxis 
                type="category" 
                dataKey="feature" 
                tick={{ fill: '#64748B', fontSize: 10 }}
                width={70}
              />
              <Bar dataKey="responsibility" radius={[0, 2, 2, 0]}>
                {insight.featureImportance.map((_entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={index === 0 ? '#00E5FF' : index === 1 ? '#FFB300' : '#64748B'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        {/* Legend */}
        <div className="flex justify-around mt-2">
          {insight.featureImportance.map((feature, index) => (
            <div key={feature.feature} className="text-center">
              <span className={`text-lg font-mono font-bold ${
                index === 0 ? 'text-[#00E5FF]' : index === 1 ? 'text-[#FFB300]' : 'text-[#64748B]'
              }`}>
                {feature.responsibility}%
              </span>
              <p className="text-[9px] text-[#64748B]">{feature.feature}</p>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
