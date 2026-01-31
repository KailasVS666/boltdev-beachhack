import { motion } from 'framer-motion';
import { GhostModel3D } from '@/components/GhostModel3D';
import { XAICard } from '@/components/XAICard';
import { ActionTerminal } from '@/components/ActionTerminal';
import type { EngineData, XAIInsight, MaintenanceAction } from '@/types/aviation';
import { ChevronLeft, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface DiagnosticViewProps {
  engineData: EngineData[];
  systemHealth: import('@/types/aviation').AircraftSystemHealth;
  xaiInsight: XAIInsight | null;
  maintenanceAction: MaintenanceAction | null;
  selectedEngine: number | null;
  onEngineSelect: (engineId: number | null) => void;
  onSubmitFeedback: (accepted: boolean, reason?: string) => void;
  onBackToDashboard: () => void;
}

export function DiagnosticView({
  engineData,
  systemHealth,
  xaiInsight,
  maintenanceAction,
  selectedEngine,
  onEngineSelect,
  onSubmitFeedback,
  onBackToDashboard
}: DiagnosticViewProps) {
  const affectedEngine = engineData.find(e => e.id === 2); // Engine 2 is the problematic one

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.6, ease: [0.34, 1.56, 0.64, 1] }}
      className="min-h-full flex flex-col lg:grid lg:grid-cols-12 gap-4 p-4"
    >
      {/* Left Column - 3D Model */}
      <div className="h-[400px] lg:h-full lg:col-span-7">
        <GhostModel3D
          engineData={engineData}
          systemHealth={systemHealth}
          selectedEngine={selectedEngine}
          onEngineSelect={onEngineSelect}
        />
      </div>

      {/* Right Column - XAI & Action Terminal */}
      <div className="flex flex-col gap-4 lg:col-span-5">
        {/* Back Button */}
        <div className="flex items-center justify-between">
          <Button
            onClick={onBackToDashboard}
            variant="outline"
            className="h-8 border-[#1E293B] text-[#64748B] hover:text-white hover:bg-[#1E293B]"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Back to Dashboard
          </Button>

          {affectedEngine && affectedEngine.vib > 2.0 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 px-3 py-1 bg-[#FF1744]/20 border border-[#FF1744]/50 rounded"
            >
              <AlertTriangle className="w-4 h-4 text-[#FF1744]" />
              <span className="text-xs font-mono text-[#FF1744]">
                ENG 2 ANOMALY DETECTED
              </span>
            </motion.div>
          )}
        </div>

        {/* XAI Card */}
        <div className="flex-1 min-h-0">
          <XAICard insight={xaiInsight} />
        </div>

        {/* Action Terminal */}
        <div className="flex-1 min-h-0">
          <ActionTerminal
            action={maintenanceAction}
            onSubmitFeedback={onSubmitFeedback}
          />
        </div>
      </div>
    </motion.div>
  );
}
