"""
AeroGuard Feedback System
Implements the "One-Way Valve" feedback loop for non-punitive learning
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config
from utils import logger, log_audit_event


class FeedbackSystem:
    """
    One-way valve feedback mechanism
    
    Key principles:
    1. Technicians can confirm or reject alerts
    2. Rejections are treated as valuable calibration data (non-punitive)
    3. Human expert review required before model retraining
    4. AeroGuard can trigger inspections, but only certified humans can sign off
    """
    
    def __init__(self, db_path: Path = config.FEEDBACK_DB):
        """
        Initialize feedback system
        
        Args:
            db_path: Path to SQLite feedback database
        """
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create feedback database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                engine_id INTEGER NOT NULL,
                aircraft_id TEXT,
                predicted_rul_mean REAL,
                predicted_rul_std REAL,
                top_sensors TEXT,
                recommended_action TEXT,
                priority TEXT,
                work_order_id TEXT,
                outcome TEXT,
                outcome_timestamp TEXT,
                technician_id TEXT,
                technician_notes TEXT,
                actual_finding TEXT,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_outcomes (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                outcome_timestamp TEXT NOT NULL,
                technician_id TEXT NOT NULL,
                technician_notes TEXT,
                actual_finding TEXT,
                parts_replaced TEXT,
                downtime_hours REAL,
                false_positive_reason TEXT,
                FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
            )
        ''')
        
        # Model retraining history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_history (
                retrain_id INTEGER PRIMARY KEY AUTOINCREMENT,
                retrain_date TEXT NOT NULL,
                model_version TEXT NOT NULL,
                num_confirmed_samples INTEGER,
                num_rejected_samples INTEGER,
                performance_metrics TEXT,
                approved_by TEXT,
                notes TEXT
            )
        ''')
        
        # Performance metrics tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                model_version TEXT,
                precision REAL,
                recall REAL,
                false_positive_rate REAL,
                false_negative_rate REAL,
                avg_rul_error REAL,
                total_alerts INTEGER,
                confirmed_alerts INTEGER,
                rejected_alerts INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback database initialized at {self.db_path}")
    
    def record_alert(
        self,
        alert: Dict,
        work_order_id: Optional[str] = None,
        model_version: str = 'v1.0'
    ):
        """
        Record issued alert to database
        
        Args:
            alert: Alert dictionary
            work_order_id: Associated work order ID
            model_version: Model version that generated the alert
        """
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (
                alert_id, timestamp, engine_id, aircraft_id,
                predicted_rul_mean, predicted_rul_std, top_sensors,
                recommended_action, priority, work_order_id,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert['alert_id'],
            alert['timestamp'],
            alert['engine_id'],
            alert.get('aircraft_id'),
            alert['rul_mean'],
            alert['rul_std'],
            json.dumps(alert['top_sensors']),
            alert['recommended_action'],
            alert['priority'],
            work_order_id,
            model_version
        ))
        
        conn.commit()
        conn.close()
        
        log_audit_event('ALERT_RECORDED', {
            'alert_id': alert['alert_id'],
            'engine_id': alert['engine_id'],
            'summary': f"Alert recorded for Engine #{alert['engine_id']}"
        })
        
        logger.info(f"Alert {alert['alert_id']} recorded to feedback database")
    
    def submit_feedback(
        self,
        alert_id: str,
        outcome: str,
        technician_id: str,
        technician_notes: Optional[str] = None,
        actual_finding: Optional[str] = None,
        parts_replaced: Optional[List[str]] = None,
        downtime_hours: Optional[float] = None,
        false_positive_reason: Optional[str] = None
    ) -> bool:
        """
        Submit technician feedback on an alert
        
        Args:
            alert_id: ID of the alert
            outcome: One of config.FEEDBACK_OUTCOMES
            technician_id: ID of technician providing feedback
            technician_notes: Free-text notes
            actual_finding: What was actually found during inspection
            parts_replaced: List of parts replaced
            downtime_hours: Actual downtime
            false_positive_reason: If rejected, why was it false positive
        
        Returns:
            Success status
        """
        if outcome not in config.FEEDBACK_OUTCOMES:
            logger.error(f"Invalid outcome: {outcome}. Must be one of {config.FEEDBACK_OUTCOMES}")
            return False
        
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update alerts table
            cursor.execute('''
                UPDATE alerts
                SET outcome = ?, outcome_timestamp = ?, technician_id = ?,
                    technician_notes = ?, actual_finding = ?
                WHERE alert_id = ?
            ''', (
                outcome,
                datetime.now().isoformat(),
                technician_id,
                technician_notes,
                actual_finding,
                alert_id
            ))
            
            # Insert into feedback_outcomes
            cursor.execute('''
                INSERT INTO feedback_outcomes (
                    alert_id, outcome, outcome_timestamp, technician_id,
                    technician_notes, actual_finding, parts_replaced,
                    downtime_hours, false_positive_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                outcome,
                datetime.now().isoformat(),
                technician_id,
                technician_notes,
                actual_finding,
                json.dumps(parts_replaced) if parts_replaced else None,
                downtime_hours,
                false_positive_reason
            ))
            
            conn.commit()
            
            log_audit_event('FEEDBACK_RECEIVED', {
                'alert_id': alert_id,
                'outcome': outcome,
                'technician_id': technician_id,
                'summary': f"Feedback submitted: {outcome}"
            }, user_id=technician_id)
            
            logger.info(f"Feedback recorded for alert {alert_id}: {outcome}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_performance_summary(
        self,
        since_date: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> Dict:
        """
        Get performance summary statistics
        
        Args:
            since_date: ISO date string (optional)
            model_version: Filter by model version (optional)
        
        Returns:
            Dictionary with performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM alerts WHERE outcome IS NOT NULL"
        params = []
        
        if since_date:
            query += " AND timestamp >= ?"
            params.append(since_date)
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) == 0:
            logger.warning("No feedback data available")
            return {
                'total_alerts': 0,
                'precision': 0.0,
                'recall': 'N/A'
            }
        
        total_alerts = len(df)
        confirmed = len(df[df['outcome'] == 'CONFIRMED'])
        rejected_fp = len(df[df['outcome'] == 'REJECTED_FALSE_POSITIVE'])
        rejected_known = len(df[df['outcome'] == 'REJECTED_ALREADY_KNOWN'])
        deferred = len(df[df['outcome'] == 'DEFERRED'])
        under_investigation = len(df[df['outcome'] == 'UNDER_INVESTIGATION'])
        
        # Precision: confirmed / (confirmed + rejected_fp)
        # Deferred and already_known are still valid alerts, just timing issue
        true_positives = confirmed + deferred + rejected_known
        false_positives = rejected_fp
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall: would need ground truth failures (not available in this design)
        
        summary = {
            'total_alerts': total_alerts,
            'confirmed': confirmed,
            'rejected_false_positive': rejected_fp,
            'rejected_already_known': rejected_known,
            'deferred': deferred,
            'under_investigation': under_investigation,
            'precision': precision,
            'false_positive_rate': false_positives / total_alerts if total_alerts > 0 else 0,
            'meets_target_precision': precision >= config.TARGET_PRECISION,
            'target_precision': config.TARGET_PRECISION
        }
        
        logger.info(f"Performance summary: {total_alerts} alerts, "
                   f"Precision: {precision:.2%} (target: {config.TARGET_PRECISION:.0%})")
        
        return summary
    
    def get_rejection_analysis(self) -> pd.DataFrame:
        """
        Analyze false positive patterns for model improvement
        
        Returns:
            DataFrame with rejection patterns
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                alert_id, engine_id, predicted_rul_mean, predicted_rul_std,
                top_sensors, recommended_action, priority,
                technician_notes, false_positive_reason
            FROM alerts
            JOIN feedback_outcomes USING (alert_id)
            WHERE outcome = 'REJECTED_FALSE_POSITIVE'
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Retrieved {len(df)} false positive cases for analysis")
        
        return df
    
    def prepare_retraining_data(
        self,
        min_samples: int = config.MIN_FEEDBACK_SAMPLES
    ) -> Tuple[bool, Dict]:
        """
        Prepare data for model retraining
        
        Checks if enough feedback is available and returns analytics
        
        Args:
            min_samples: Minimum feedback samples required
        
        Returns:
            (is_ready, analytics_dict)
        """
        performance = self.get_performance_summary()
        
        total_feedback = (
            performance['confirmed'] +
            performance['rejected_false_positive'] +
            performance['rejected_already_known'] +
            performance['deferred']
        )
        
        is_ready = total_feedback >= min_samples
        
        analytics = {
            'total_feedback_samples': total_feedback,
            'min_required': min_samples,
            'is_ready_for_retraining': is_ready,
            'performance_summary': performance
        }
        
        if is_ready:
            logger.info(f"✓ Ready for retraining: {total_feedback} samples (min: {min_samples})")
        else:
            logger.info(f"✗ Not ready for retraining: {total_feedback}/{min_samples} samples")
        
        return is_ready, analytics
    
    def record_retraining(
        self,
        model_version: str,
        num_confirmed: int,
        num_rejected: int,
        performance_metrics: Dict,
        approved_by: str,
        notes: Optional[str] = None
    ):
        """
        Record model retraining event
        
        Args:
            model_version: New model version
            num_confirmed: Number of confirmed samples used
            num_rejected: Number of rejected samples used
            performance_metrics: Validation metrics
            approved_by: ID of approving expert
            notes: Optional notes
        """
        import json
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO retraining_history (
                retrain_date, model_version, num_confirmed_samples,
                num_rejected_samples, performance_metrics, approved_by, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            model_version,
            num_confirmed,
            num_rejected,
            json.dumps(performance_metrics),
            approved_by,
            notes
        ))
        
        conn.commit()
        conn.close()
        
        log_audit_event('MODEL_RETRAINED', {
            'model_version': model_version,
            'num_samples': num_confirmed + num_rejected,
            'approved_by': approved_by
        }, user_id=approved_by)
        
        logger.info(f"Retraining event recorded: {model_version} by {approved_by}")
    
    def export_feedback_for_review(self, output_path: Path) -> Path:
        """
        Export feedback data for human expert review
        
        Args:
            output_path: Path to save CSV file
        
        Returns:
            Path to exported file
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                a.alert_id, a.timestamp, a.engine_id, a.aircraft_id,
                a.predicted_rul_mean, a.predicted_rul_std, a.priority,
                a.outcome, f.technician_notes, f.actual_finding,
                f.false_positive_reason, a.model_version
            FROM alerts a
            LEFT JOIN feedback_outcomes f ON a.alert_id = f.alert_id
            WHERE a.outcome IS NOT NULL
            ORDER BY a.timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df.to_csv(output_path, index=False)
        
        logger.info(f"Feedback data exported to {output_path} ({len(df)} records)")
        
        return output_path


# ==================== WORKFLOW FUNCTIONS ====================
def simulate_maintenance_workflow(
    alert: Dict,
    work_order: Dict,
    feedback_system: FeedbackSystem,
    technician_id: str = "TECH-001"
) -> Dict:
    """
    Simulate complete maintenance workflow with feedback
    
    This demonstrates the "one-way valve" in action
    
    Args:
        alert: Alert from explainability engine
        work_order: Work order from maintenance integration
        feedback_system: Feedback system instance
        technician_id: Technician ID
    
    Returns:
        Workflow summary
    """
    logger.info("=" * 70)
    logger.info("SIMULATING MAINTENANCE WORKFLOW")
    logger.info("=" * 70)
    
    # Step 1: Record alert
    feedback_system.record_alert(alert, work_order['work_order_id'])
    print(f"\n✓ Alert {alert['alert_id']} issued to maintenance team")
    
    # Step 2: Technician reviews alert
    print(f"✓ Technician {technician_id} reviews alert")
    print(f"  Priority: {alert['priority']}")
    print(f"  Recommended: {alert['recommended_action'][:80]}...")
    
    # Step 3: Technician performs inspection
    # For demo, randomly decide outcome
    import random
    outcome_choice = random.choice([
        'CONFIRMED',
        'REJECTED_FALSE_POSITIVE',
        'REJECTED_ALREADY_KNOWN'
    ])
    
    print(f"\n✓ Inspection performed")
    print(f"  Outcome: {outcome_choice}")
    
    # Step 4: Submit feedback
    if outcome_choice == 'CONFIRMED':
        feedback_system.submit_feedback(
            alert['alert_id'],
            outcome='CONFIRMED',
            technician_id=technician_id,
            technician_notes="Alert was accurate. Found degradation as predicted.",
            actual_finding="LPT blade erosion detected. Replaced blade set.",
            parts_replaced=["LPT-BLADE-001"],
            downtime_hours=48.0
        )
    elif outcome_choice == 'REJECTED_FALSE_POSITIVE':
        feedback_system.submit_feedback(
            alert['alert_id'],
            outcome='REJECTED_FALSE_POSITIVE',
            technician_id=technician_id,
            technician_notes="No issue found. Sensors readings were within normal limits.",
            false_positive_reason="Environmental conditions (high altitude operations) triggered alert"
        )
    else:  # REJECTED_ALREADY_KNOWN
        feedback_system.submit_feedback(
            alert['alert_id'],
            outcome='REJECTED_ALREADY_KNOWN',
            technician_id=technician_id,
            technician_notes="Issue was already identified in previous inspection.",
            actual_finding="Known LPT wear, already scheduled for next maintenance window"
        )
    
    print(f"✓ Feedback submitted (non-punitive, stored for learning)")
    
    # Step 5: Check if ready for retraining
    is_ready, analytics = feedback_system.prepare_retraining_data()
    
    workflow_summary = {
        'alert_id': alert['alert_id'],
        'work_order_id': work_order['work_order_id'],
        'outcome': outcome_choice,
        'technician_id': technician_id,
        'retraining_ready': is_ready,
        'performance_summary': feedback_system.get_performance_summary()
    }
    
    logger.info("=" * 70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 70)
    
    return workflow_summary


if __name__ == '__main__':
    # Demo
    print("=" * 70)
    print("AeroGuard Feedback System - Demo")
    print("=" * 70)
    
    feedback = FeedbackSystem()
    
    # Simulate some feedback
    for i in range(5):
        mock_alert = {
            'alert_id': f'AG-TEST-{i}',
            'timestamp': datetime.now().isoformat(),
            'engine_id': i % 3 + 1,
            'rul_mean': 50.0,
            'rul_std': 10.0,
            'top_sensors': [('T50', 0.4), ('Nc', 0.3)],
            'recommended_action': 'Inspect LPT',
            'priority': 'MEDIUM'
        }
        
        feedback.record_alert(mock_alert, f'WO-TEST-{i}')
        
        # Simulate feedback
        outcomes = ['CONFIRMED', 'REJECTED_FALSE_POSITIVE', 'DEFERRED']
        feedback.submit_feedback(
            f'AG-TEST-{i}',
            outcomes[i % 3],
            'TECH-001',
            'Test feedback'
        )
    
    # Get performance summary
    summary = feedback.get_performance_summary()
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total Alerts: {summary['total_alerts']}")
    print(f"Confirmed: {summary['confirmed']}")
    print(f"False Positives: {summary['rejected_false_positive']}")
    print(f"Precision: {summary['precision']:.1%} (Target: {summary['target_precision']:.0%})")
    print(f"Meets Target: {'✓ YES' if summary['meets_target_precision'] else '✗ NO'}")
    
    print("\n" + "=" * 70)
    print("Feedback System Ready")
    print("=" * 70)
