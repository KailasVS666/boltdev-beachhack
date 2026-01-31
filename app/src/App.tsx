import { useFlightData } from '@/hooks/useFlightData';
import { VitalsStrip } from '@/components/VitalsStrip';
import { DashboardView } from '@/views/DashboardView';
import { DiagnosticView } from '@/views/DiagnosticView';
import { AnimatePresence } from 'framer-motion';
import { Play, Pause, Maximize2, Minimize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import './App.css';

function App() {
  const {
    currentRow,
    engineData,
    systemHealth,
    dqi,
    systemState,
    viewMode,
    setViewMode,
    selectedEngine,
    setSelectedEngine,
    xaiInsight,
    maintenanceAction,
    isPlaying,
    togglePlayback,
    currentRowIndex,
    totalRows,
    submitFeedback,
    historicalData
  } = useFlightData();

  return (
    <div className={`h-screen bg-[#050810] flex flex-col ${systemState.status === 'crisis' ? 'crisis-pulse' :
      systemState.status === 'alert' ? 'amber-pulse' : ''
      }`}>
      {/* Vitals Strip Header */}
      <VitalsStrip
        currentRow={currentRow}
        dqi={dqi}
        systemStatus={systemState.status}
      />

      {/* Main Content Area */}
      <main className="flex-1 overflow-auto relative scrollable">
        <AnimatePresence mode="wait">
          {viewMode === 'dashboard' ? (
            <DashboardView
              key="dashboard"
              engineData={engineData}
              historicalData={historicalData}
              currentRow={currentRow}
            />
          ) : (
            <DiagnosticView
              key="diagnostic"
              engineData={engineData}
              systemHealth={systemHealth!}
              xaiInsight={xaiInsight}
              maintenanceAction={maintenanceAction}
              selectedEngine={selectedEngine}
              onEngineSelect={setSelectedEngine}
              onSubmitFeedback={submitFeedback}
              onBackToDashboard={() => setViewMode('dashboard')}
            />
          )}
        </AnimatePresence>
      </main>

      {/* Control Bar */}
      <footer className="h-12 bg-[#050810] border-t border-[#1E293B] flex items-center px-6 relative">
        {/* Playback Controls */}
        <div className="flex items-center gap-4">
          <Button
            onClick={togglePlayback}
            variant="outline"
            size="sm"
            className="h-8 border-[#1E293B] text-[#64748B] hover:text-white hover:bg-[#1E293B]"
          >
            {isPlaying ? (
              <Pause className="w-4 h-4 mr-1" />
            ) : (
              <Play className="w-4 h-4 mr-1" />
            )}
            {isPlaying ? 'PAUSE' : 'PLAY'}
          </Button>

          {/* Progress Indicator */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#64748B] font-mono">
              ROW {currentRowIndex + 1} / {totalRows}
            </span>
            <div className="w-32 h-1 bg-[#1E293B] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#00E5FF] transition-all duration-300"
                style={{ width: `${((currentRowIndex + 1) / totalRows) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* View Toggle - Centered */}
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2">
          <Button
            onClick={() => setViewMode('dashboard')}
            variant={viewMode === 'dashboard' ? 'default' : 'outline'}
            size="sm"
            className={`h-8 ${viewMode === 'dashboard'
              ? 'bg-[#00E5FF] text-black hover:bg-[#00B8D4]'
              : 'border-[#1E293B] text-[#64748B] hover:text-white'
              }`}
          >
            <Minimize2 className="w-4 h-4 mr-1" />
            DASHBOARD
          </Button>
          <Button
            onClick={() => setViewMode('diagnostic')}
            variant={viewMode === 'diagnostic' ? 'default' : 'outline'}
            size="sm"
            className={`h-8 ${viewMode === 'diagnostic'
              ? 'bg-[#00E5FF] text-black hover:bg-[#00B8D4]'
              : 'border-[#1E293B] text-[#64748B] hover:text-white'
              }`}
          >
            <Maximize2 className="w-4 h-4 mr-1" />
            3D DIAGNOSTIC
          </Button>
        </div>

        {/* System Status */}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-[10px] text-[#64748B]">STATUS</span>
          <div className={`px-2 py-1 rounded text-[10px] font-mono font-bold ${systemState.status === 'nominal' ? 'bg-[#00E5FF]/20 text-[#00E5FF]' :
            systemState.status === 'alert' ? 'bg-[#FFB300]/20 text-[#FFB300]' :
              'bg-[#FF1744]/20 text-[#FF1744]'
            }`}>
            {systemState.status.toUpperCase()}
          </div>
        </div>
      </footer>

      {/* Alert Overlay */}
      {systemState.status === 'crisis' && (
        <div className="absolute inset-0 pointer-events-none flex items-start justify-center pt-20 z-40">
          <div className="bg-[#FF1744]/90 text-white px-6 py-3 rounded border-2 border-[#FF1744] animate-pulse">
            <span className="text-lg font-bold font-mono tracking-wider">
              ⚠ CRITICAL ALERT: {systemState.message}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
