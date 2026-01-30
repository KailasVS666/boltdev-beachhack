"""
AeroGuard Maintenance Integration
Convert predictions into actionable draft work orders with parts inventory checks
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import config
from utils import logger, calculate_risk_envelope, log_audit_event


class MaintenanceIntegration:
    """
    Maintenance Operations interface for AeroGuard
    
    Features:
    1. Risk envelope calculation (safe flight budget)
    2. Parts inventory checking (mock)
    3. Draft work order generation
    4. Priority scoring
    """
    
    def __init__(self, parts_db_path: Optional[Path] = None):
        """
        Initialize maintenance integration
        
        Args:
            parts_db_path: Path to parts inventory database (mock for demo)
        """
        self.parts_db_path = parts_db_path or (config.BASE_DIR / "parts_inventory.db")
        self._initialize_mock_parts_db()
    
    def _initialize_mock_parts_db(self):
        """Create mock parts inventory database"""
        conn = sqlite3.connect(self.parts_db_path)
        cursor = conn.cursor()
        
        # Create parts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parts_inventory (
                part_number TEXT PRIMARY KEY,
                description TEXT,
                quantity_in_stock INTEGER,
                lead_time_days INTEGER,
                cost REAL,
                category TEXT
            )
        ''')
        
        # Insert sample parts
        sample_parts = [
            ('LPT-BLADE-001', 'LPT Blade Set', 3, 14, 45000.0, 'Turbine'),
            ('LPT-NGV-001', 'LPT Nozzle Guide Vanes', 5, 10, 28000.0, 'Turbine'),
            ('HPC-BLADE-001', 'HPC Blade Set', 2, 21, 52000.0, 'Compressor'),
            ('HPC-SEAL-001', 'HPC Seals Kit', 8, 7, 3500.0, 'Seals'),
            ('BEARING-FAN-001', 'Fan Bearing Set', 4, 14, 12000.0, 'Bearings'),
            ('BEARING-CORE-001', 'Core Rotor Bearings', 3, 18, 18000.0, 'Bearings'),
            ('FUEL-NOZZLE-001', 'Fuel Nozzle Set', 6, 5, 8000.0, 'Fuel System'),
            ('SEAL-KIT-001', 'Pressure Seals and Gaskets', 12, 3, 1200.0, 'Seals'),
        ]
        
        cursor.executemany(
            'INSERT OR IGNORE INTO parts_inventory VALUES (?, ?, ?, ?, ?, ?)',
            sample_parts
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Mock parts inventory initialized at {self.parts_db_path}")
    
    def calculate_flight_budget(
        self,
        rul_mean: float,
        rul_std: float,
        confidence_level: str = config.DEFAULT_CONFIDENCE_LEVEL
    ) -> Dict[str, float]:
        """
        Calculate safe operating window with FAA safety margins
        
        Args:
            rul_mean: Mean predicted RUL
            rul_std: Standard deviation of RUL
            confidence_level: 'conservative' | 'standard' | 'optimistic'
        
        Returns:
            Dictionary with flight budget metrics
        """
        confidence = config.CONFIDENCE_LEVELS[confidence_level]
        
        # Base risk envelope
        risk_envelope = calculate_risk_envelope(
            rul_mean, rul_std, target_flights=50, confidence_level=confidence
        )
        
        # Apply FAA safety margin (reduce by 20% for conservative ops)
        safe_budget = risk_envelope['safe_flight_budget']
        safe_budget_with_margin = safe_budget * (1 - config.FAA_SAFETY_MARGIN)
        
        result = {
            'safe_flight_budget': safe_budget_with_margin,
            'safe_budget_no_margin': safe_budget,
            'rul_mean': rul_mean,
            'rul_std': rul_std,
            'confidence': confidence,
            'uncertainty_ratio': risk_envelope['uncertainty_ratio'],
            'faa_margin_applied': config.FAA_SAFETY_MARGIN
        }
        
        logger.info(f"Flight budget calculated: {safe_budget_with_margin:.0f} flights "
                   f"(with {config.FAA_SAFETY_MARGIN*100:.0f}% FAA safety margin)")
        
        return result
    
    def check_parts_availability(self, parts_list: List[str]) -> List[Dict]:
        """
        Check inventory for required parts
        
        Args:
            parts_list: List of part descriptions
        
        Returns:
            List of part availability status
        """
        conn = sqlite3.connect(self.parts_db_path)
        cursor = conn.cursor()
        
        availability = []
        
        for part_desc in parts_list:
            # Simple keyword matching for demo
            # In production, would use proper part number lookup
            keywords = part_desc.lower().split()
            
            cursor.execute('''
                SELECT part_number, description, quantity_in_stock, lead_time_days, cost
                FROM parts_inventory
                WHERE LOWER(description) LIKE ?
                LIMIT 1
            ''', (f'%{keywords[0]}%',))
            
            result = cursor.fetchone()
            
            if result:
                part_number, description, qty, lead_time, cost = result
                in_stock = qty > 0
                
                availability.append({
                    'part_description': part_desc,
                    'part_number': part_number,
                    'in_stock': in_stock,
                    'quantity': qty,
                    'lead_time_days': lead_time if not in_stock else 0,
                    'cost': cost,
                    'status': 'AVAILABLE' if in_stock else f'ORDER REQUIRED ({lead_time} days)'
                })
            else:
                availability.append({
                    'part_description': part_desc,
                    'part_number': 'TBD',
                    'in_stock': False,
                    'quantity': 0,
                    'lead_time_days': config.PARTS_LEAD_TIME_DAYS,
                    'cost': 0.0,
                    'status': 'PART NUMBER TO BE DETERMINED'
                })
        
        conn.close()
        
        return availability
    
    def calculate_priority_score(
        self,
        rul_mean: float,
        rul_std: float,
        flights_scheduled_next_week: int,
        parts_availability: List[Dict]
    ) -> Tuple[float, str]:
        """
        Calculate priority score for work order
        
        Args:
            rul_mean: Mean RUL
            rul_std: RUL uncertainty
            flights_scheduled_next_week: Number of flights planned
            parts_availability: Parts availability status
        
        Returns:
            (priority_score, priority_level)
        """
        weights = config.PRIORITY_WEIGHTS
        
        # 1. Risk score (0-1): Higher if RUL is low
        risk_score = 1.0 - min(rul_mean / config.RUL_MEDIUM, 1.0)
        
        # 2. Uncertainty penalty (0-1): Higher if uncertainty is high
        uncertainty_ratio = rul_std / (rul_mean + 1e-8)
        uncertainty_score = min(uncertainty_ratio / config.MAX_ACCEPTABLE_UNCERTAINTY, 1.0)
        
        # 3. Flight schedule urgency (0-1): Higher if many flights scheduled
        schedule_score = min(flights_scheduled_next_week / 10.0, 1.0)
        
        # 4. Parts availability (0-1): Higher if parts are in stock
        parts_in_stock = sum(1 for p in parts_availability if p['in_stock'])
        parts_score = parts_in_stock / len(parts_availability) if parts_availability else 0.5
        
        # Weighted sum
        priority_score = (
            weights['risk_score'] * risk_score +
            weights['uncertainty'] * uncertainty_score +
            weights['flight_schedule'] * schedule_score +
            weights['parts_availability'] * parts_score
        )
        
        # Determine priority level
        if priority_score > 0.75 or rul_mean < config.RUL_CRITICAL:
            priority_level = 'CRITICAL'
        elif priority_score > 0.6 or rul_mean < config.RUL_HIGH:
            priority_level = 'HIGH'
        elif priority_score > 0.4 or rul_mean < config.RUL_MEDIUM:
            priority_level = 'MEDIUM'
        else:
            priority_level = 'LOW'
        
        return priority_score, priority_level
    
    def generate_work_order(
        self,
        alert: Dict,
        aircraft_id: str,
        flights_scheduled: int = 5,
        mechanic_id: Optional[str] = None
    ) -> Dict:
        """
        Generate draft work order from alert
        
        Args:
            alert: Alert dictionary from explainability engine
            aircraft_id: Aircraft registration number
            flights_scheduled: Flights scheduled in next 7 days
            mechanic_id: Assigned mechanic (if known)
        
        Returns:
            Work order dictionary
        """
        logger.info(f"Generating work order for Engine #{alert['engine_id']}")
        
        # Calculate flight budget
        flight_budget = self.calculate_flight_budget(
            alert['rul_mean'],
            alert['rul_std'],
            alert['confidence_level']
        )
        
        # Check parts availability
        parts_needs = self.check_parts_availability(alert['parts_list'])
        
        # Calculate priority
        priority_score, priority_level = self.calculate_priority_score(
            alert['rul_mean'],
            alert['rul_std'],
            flights_scheduled,
            parts_needs
        )
        
        # Calculate deadline
        safe_flights = int(flight_budget['safe_flight_budget'])
        # Assume average 2 flights per day
        days_until_action = max(safe_flights // 2, 1)
        deadline = datetime.now() + timedelta(days=days_until_action)
        
        # Generate work order ID
        wo_id = f"WO-{aircraft_id}-{alert['alert_id']}"
        
        work_order = {
            'work_order_id': wo_id,
            'created_at': datetime.now().isoformat(),
            'aircraft_id': aircraft_id,
            'engine_id': alert['engine_id'],
            'alert_id': alert['alert_id'],
            'priority': priority_level,
            'priority_score': float(priority_score),
            'priority_description': config.WORK_ORDER_PRIORITY[priority_level],
            
            # RUL predictions
            'predicted_rul_mean': float(alert['rul_mean']),
            'predicted_rul_std': float(alert['rul_std']),
            'safe_flight_budget': float(flight_budget['safe_flight_budget']),
            'confidence_level': alert['confidence_level'],
            
            # Maintenance details
            'recommended_action': alert['recommended_action'],
            'top_sensors': alert['top_sensors'],
            
            # Parts
            'parts_needed': parts_needs,
            'total_parts_cost': sum(p['cost'] for p in parts_needs),
            
            # Scheduling
            'deadline': deadline.isoformat(),
            'flights_scheduled_next_week': flights_scheduled,
            'estimated_downtime_hours': self._estimate_downtime(alert['recommended_action']),
            
            # Workflow
            'status': 'DRAFT',
            'assigned_to': mechanic_id,
            'requires_signoff': config.REQUIRE_HUMAN_SIGNOFF,
            'advisory_only': config.ADVISORY_ONLY_MODE,
            
            # Documentation
            'alert_message': alert['alert_message'],
            'created_by': 'AeroGuard_AI_System'
        }
        
        # Log to audit trail
        log_audit_event(
            'WORK_ORDER_GENERATED',
            {
                'work_order_id': wo_id,
                'aircraft_id': aircraft_id,
                'priority': priority_level,
                'summary': f"Work order for Engine #{alert['engine_id']} - RUL {alert['rul_mean']:.0f}±{alert['rul_std']:.0f}"
            }
        )
        
        logger.info(f"Work order {wo_id} generated - Priority: {priority_level}, "
                   f"Deadline: {deadline.strftime('%Y-%m-%d')}")
        
        return work_order
    
    def _estimate_downtime(self, recommended_action: str) -> int:
        """
        Estimate maintenance downtime in hours
        
        Args:
            recommended_action: Description of required actions
        
        Returns:
            Estimated hours
        """
        # Simple keyword-based estimation
        action_lower = recommended_action.lower()
        
        if 'borescope' in action_lower:
            return 4  # Borescope inspection: 4 hours
        elif 'blade' in action_lower and 'replace' in action_lower:
            return 48  # Blade replacement: 48 hours
        elif 'bearing' in action_lower:
            return 24  # Bearing replacement: 24 hours
        elif 'pressure test' in action_lower:
            return 6  # Pressure testing: 6 hours
        elif 'fuel' in action_lower:
            return 8  # Fuel system work: 8 hours
        else:
            return 12  # Default inspection: 12 hours
    
    def format_work_order_for_print(self, work_order: Dict) -> str:
        """
        Format work order as human-readable document
        
        Args:
            work_order: Work order dictionary
        
        Returns:
            Formatted string
        """
        wo = work_order
        
        doc = f"""
╔════════════════════════════════════════════════════════════════════════
║                    DRAFT MAINTENANCE WORK ORDER
║                    (AI-Generated - Requires Human Approval)
╠════════════════════════════════════════════════════════════════════════
║ Work Order ID: {wo['work_order_id']}
║ Created: {datetime.fromisoformat(wo['created_at']).strftime('%Y-%m-%d %H:%M:%S')}
║ 
║ AIRCRAFT INFORMATION:
║   Aircraft ID: {wo['aircraft_id']}
║   Engine Unit: #{wo['engine_id']}
║
║ PRIORITY: {wo['priority']} (Score: {wo['priority_score']:.2f})
║   {wo['priority_description']}
║   Deadline: {datetime.fromisoformat(wo['deadline']).strftime('%Y-%m-%d')}
║
║ PREDICTIVE ANALYSIS:
║   Remaining Useful Life: {wo['predicted_rul_mean']:.0f} ± {wo['predicted_rul_std']:.0f} cycles
║   Safe Flight Budget: {wo['safe_flight_budget']:.0f} flights (with FAA margin)
║   Confidence: {wo['confidence_level']}
║
║ PRIMARY SENSOR INDICATORS:
"""
        
        for sensor, contrib in wo['top_sensors']:
            doc += f"║   • {sensor}: {contrib*100:.1f}% contribution\n"
        
        doc += f"""║
║ RECOMMENDED ACTIONS:
║   {wo['recommended_action']}
║
║ PARTS REQUIRED:
"""
        
        for part in wo['parts_needed']:
            doc += f"║   • {part['part_description']}\n"
            doc += f"║     PN: {part['part_number']} | Status: {part['status']}\n"
        
        doc += f"""║
║   Total Estimated Parts Cost: ${wo['total_parts_cost']:,.2f}
║
║ ESTIMATED DOWNTIME: {wo['estimated_downtime_hours']} hours
║
║ STATUS: {wo['status']}
║ Assigned To: {wo['assigned_to'] or 'Unassigned'}
║
║ ⚠️  IMPORTANT: This work order was generated by AeroGuard AI system.
║     All maintenance actions must be verified by a certified technician
║     per applicable maintenance manual and regulatory requirements.
║     Technician signature required before execution.
╚════════════════════════════════════════════════════════════════════════
"""
        
        return doc


if __name__ == '__main__':
    # Demo
    print("=" * 70)
    print("AeroGuard Maintenance Integration - Demo")
    print("=" * 70)
    
    integration = MaintenanceIntegration()
    
    # Mock alert
    mock_alert = {
        'alert_id': 'AG-3-20260130110000',
        'engine_id': 3,
        'rul_mean': 45.0,
        'rul_std': 8.5,
        'confidence_level': 'conservative',
        'top_sensors': [
            ('T50_LPT_Temp', 0.35),
            ('Nc_Core_Speed', 0.22)
        ],
        'recommended_action': 'Borescope inspection of LPT blades | Inspect core rotor bearings',
        'parts_list': ['LPT blade set (PN: TBD)', 'Core rotor bearings (PN: TBD)'],
        'alert_message': '[Alert message]'
    }
    
    # Generate work order
    work_order = integration.generate_work_order(
        mock_alert,
        aircraft_id='N12345',
        flights_scheduled=7
    )
    
    print("\n" + integration.format_work_order_for_print(work_order))
    
    print("\n" + "=" * 70)
    print("Maintenance Integration Ready")
    print("=" * 70)
