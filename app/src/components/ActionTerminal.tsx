import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { MaintenanceAction } from '@/types/aviation';
import { Terminal, Check, X, FileText, Package, DollarSign, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { mockInventory } from '@/data/mockFlightData';

interface ActionTerminalProps {
  action: MaintenanceAction | null;
  onSubmitFeedback: (accepted: boolean, reason?: string) => void;
}

export function ActionTerminal({ action, onSubmitFeedback }: ActionTerminalProps) {
  const [showRejectForm, setShowRejectForm] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  
  if (!action) {
    return (
      <div className="terminal-window p-6 h-full flex flex-col items-center justify-center">
        <Terminal className="w-12 h-12 text-[#1E293B] mb-4" />
        <p className="text-sm text-[#64748B]">No maintenance actions pending</p>
        <p className="text-[10px] text-[#475569] mt-2">System monitoring for anomalies...</p>
      </div>
    );
  }
  
  const handleAccept = () => {
    onSubmitFeedback(true);
  };
  
  const handleReject = () => {
    if (showRejectForm && rejectReason.trim()) {
      onSubmitFeedback(false, rejectReason);
      setShowRejectForm(false);
      setRejectReason('');
    } else {
      setShowRejectForm(true);
    }
  };
  
  const inventory = mockInventory[action.partNumber as keyof typeof mockInventory];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="terminal-window p-6 h-full"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-sm font-bold text-white tracking-wider flex items-center gap-2">
          <span className="w-1 h-4 bg-[#FF1744] rounded-full" />
          ACTION TERMINAL
        </h2>
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#FF1744]" />
          <span className="text-[10px] text-[#64748B]">MAINTENANCE REQUEST</span>
        </div>
      </div>
      
      {/* AMM Reference */}
      <div className="mb-4 p-3 bg-[#0B1120] rounded border border-[#1E293B]">
        <div className="flex items-center gap-2 mb-1">
          <FileText className="w-4 h-4 text-[#00E5FF]" />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">AMM Reference</span>
        </div>
        <p className="text-lg font-mono font-bold text-[#00E5FF]">
          {action.ammReference}
        </p>
      </div>
      
      {/* Part Information */}
      <div className="mb-4 p-3 bg-[#0B1120] rounded border border-[#1E293B]">
        <div className="flex items-center gap-2 mb-2">
          <Package className="w-4 h-4 text-[#FFB300]" />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Part Information</span>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-[#64748B]">Part #</span>
            <span className="text-xs font-mono text-white">{action.partNumber}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-xs text-[#64748B]">Name</span>
            <span className="text-xs text-white">{action.partName}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-xs text-[#64748B]">Stock</span>
            <span className={`text-xs font-mono font-bold ${
              action.stockCount > 0 ? 'text-[#00E5FF]' : 'text-[#FF1744]'
            }`}>
              {action.stockCount} {action.stockCount === 1 ? 'unit' : 'units'}
            </span>
          </div>
          {inventory && (
            <div className="flex justify-between">
              <span className="text-xs text-[#64748B]">Location</span>
              <span className="text-xs text-[#94A3B8]">{inventory.location}</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Cost Estimate */}
      <div className="mb-6 p-3 bg-[#0B1120] rounded border border-[#1E293B]">
        <div className="flex items-center gap-2 mb-1">
          <DollarSign className="w-4 h-4 text-[#00E5FF]" />
          <span className="text-[10px] text-[#64748B] uppercase tracking-wider">Estimated Cost</span>
        </div>
        <p className="text-xl font-mono font-bold text-white">
          ${action.estimatedCost.toLocaleString()}
        </p>
      </div>
      
      {/* Human Gate - Action Buttons */}
      <AnimatePresence mode="wait">
        {!showRejectForm ? (
          <motion.div
            key="buttons"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            <p className="text-[10px] text-[#64748B] uppercase tracking-wider mb-2">
              Human Verification Required
            </p>
            
            <Button
              onClick={handleAccept}
              className="w-full h-12 bg-[#00E5FF] hover:bg-[#00B8D4] text-black font-bold tracking-wider"
            >
              <Check className="w-4 h-4 mr-2" />
              ACCEPT & PLAN
            </Button>
            
            <Button
              onClick={handleReject}
              variant="outline"
              className="w-full h-12 border-[#FF1744] text-[#FF1744] hover:bg-[#FF1744]/10 font-bold tracking-wider"
            >
              <X className="w-4 h-4 mr-2" />
              REJECT - SENSOR GLITCH
            </Button>
          </motion.div>
        ) : (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            <div className="flex items-center gap-2 mb-2">
              <MessageSquare className="w-4 h-4 text-[#FF1744]" />
              <span className="text-[10px] text-[#64748B] uppercase tracking-wider">
                Why is the AI wrong?
              </span>
            </div>
            
            <Textarea
              value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="Enter your reasoning..."
              className="bg-[#0B1120] border-[#1E293B] text-white text-sm min-h-[80px] resize-none"
            />
            
            <div className="flex gap-2">
              <Button
                onClick={handleReject}
                disabled={!rejectReason.trim()}
                className="flex-1 h-10 bg-[#FF1744] hover:bg-[#D50000] text-white font-bold"
              >
                SUBMIT REJECTION
              </Button>
              <Button
                onClick={() => {
                  setShowRejectForm(false);
                  setRejectReason('');
                }}
                variant="outline"
                className="h-10 border-[#1E293B] text-[#64748B]"
              >
                CANCEL
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Footer Note */}
      <div className="mt-4 pt-4 border-t border-[#1E293B]">
        <p className="text-[9px] text-[#475569] text-center">
          Feedback will be saved to engineer_feedback.csv for model retraining
        </p>
      </div>
    </motion.div>
  );
}
