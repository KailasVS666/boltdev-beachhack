import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, AlertCircle, Info, X, ChevronDown, ChevronUp, Bell } from 'lucide-react';
import type { SystemError } from '@/types/aviation';
import { useState } from 'react';

interface ErrorBannerProps {
    errors: SystemError[];
}

export function ErrorBanner({ errors }: ErrorBannerProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [dismissedErrors, setDismissedErrors] = useState<Set<string>>(new Set());

    if (errors.length === 0) return null;

    const visibleErrors = errors.filter(
        error => !dismissedErrors.has(`${error.category}-${error.message}`)
    );

    if (visibleErrors.length === 0) return null;

    // Count breakdown
    const criticalCount = visibleErrors.filter(e => e.priority === 'CRITICAL' || e.severity === 'critical').length;
    const urgentCount = visibleErrors.filter(e => e.priority === 'URGENT').length;
    const watchlistCount = visibleErrors.filter(e => e.priority === 'WATCHLIST').length;

    const topPriority = criticalCount > 0 ? 'CRITICAL' : urgentCount > 0 ? 'URGENT' : 'WATCHLIST';

    return (
        <div className="fixed top-20 right-4 z-50 flex flex-col items-end gap-2 max-w-md pointer-events-auto">
            {/* Summary Button / Toggle */}
            <motion.button
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                onClick={() => setIsExpanded(!isExpanded)}
                className={`
                    flex items-center gap-3 px-4 py-2 rounded-full shadow-lg backdrop-blur-md border transition-all
                    ${topPriority === 'CRITICAL' ? 'bg-red-950/90 border-red-500 text-red-100' :
                        topPriority === 'URGENT' ? 'bg-orange-950/90 border-orange-500 text-orange-100' :
                            'bg-amber-950/90 border-amber-500 text-amber-100'}
                `}
            >
                <div className="relative">
                    <Bell className="w-4 h-4" />
                    <span className="absolute -top-1 -right-1 flex h-2 w-2">
                        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${topPriority === 'CRITICAL' ? 'bg-red-400' : 'bg-orange-400'}`}></span>
                        <span className={`relative inline-flex rounded-full h-2 w-2 ${topPriority === 'CRITICAL' ? 'bg-red-500' : 'bg-orange-500'}`}></span>
                    </span>
                </div>
                <span className="text-sm font-bold tracking-wide">
                    {visibleErrors.length} Active Alert{visibleErrors.length !== 1 ? 's' : ''}
                </span>
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </motion.button>

            {/* Expandable List */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: -10 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className="flex flex-col gap-2 w-full mt-2 lg:w-96 max-h-[60vh] overflow-y-auto pr-1"
                    >
                        {visibleErrors.map((error, index) => {
                            const isCritical = error.priority === 'CRITICAL' || error.severity === 'critical';
                            const isUrgent = error.priority === 'URGENT';
                            const isWatchlist = error.priority === 'WATCHLIST';

                            let bgColor, borderColor, iconColor, Icon;

                            if (isCritical) {
                                bgColor = 'bg-red-950/95';
                                borderColor = 'border-red-500';
                                iconColor = 'text-red-400';
                                Icon = AlertCircle;
                            } else if (isUrgent) {
                                bgColor = 'bg-orange-950/95';
                                borderColor = 'border-orange-500';
                                iconColor = 'text-orange-400';
                                Icon = AlertTriangle;
                            } else {
                                bgColor = 'bg-amber-950/95';
                                borderColor = 'border-amber-500';
                                iconColor = 'text-amber-400';
                                Icon = Info;
                            }

                            return (
                                <motion.div
                                    key={`${error.category}-${index}`}
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className={`${bgColor} ${borderColor} border-l-4 rounded-r-lg p-3 backdrop-blur-sm shadow-xl relative group`}
                                >
                                    <div className="flex items-start gap-3">
                                        <Icon className={`w-4 h-4 ${iconColor} flex-shrink-0 mt-1`} />

                                        <div className="flex-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className={`text-[10px] font-bold ${iconColor} uppercase tracking-wider`}>
                                                    {error.priority || error.severity}
                                                </span>
                                                {error.ammReference && (
                                                    <span className="text-[9px] text-gray-500 font-mono">
                                                        {error.ammReference}
                                                    </span>
                                                )}
                                            </div>

                                            <p className="text-xs text-white font-medium leading-tight">
                                                {error.message}
                                            </p>

                                            {error.value !== undefined && (
                                                <div className="mt-1 flex items-center gap-2 text-[10px] text-gray-400 font-mono">
                                                    <span>Val: <span className="text-white">{error.value.toFixed(1)}</span></span>
                                                    <span>/</span>
                                                    <span>Lim: {error.threshold?.toFixed(1)}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
