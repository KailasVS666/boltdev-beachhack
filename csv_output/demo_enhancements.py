"""
Quick Demo: Enhanced AeroGuard Features
Shows the 3 key improvements over base model
"""

import sys
sys.path.append('.')
from enhanced_inference import EnhancedAeroGuard

def quick_demo():
    print("ENHANCED AEROGUARD - Quick Demo")
    print("="*60)
    
    engine = EnhancedAeroGuard()
    
    # Test flight data
    flight = {
        'ALT': 800, 'VRTG': 1.8, 'GS': 180,
        'EGT_1': 920, 'N1_1': 9200, 'VIB_1': 2.5,
        'FF_1': 4500, 'OIT_1': 95
    }
    
    print("\nPredicting RUL for flight...")
    result = engine.predict_flight(flight)
    
    print(f"\nRUL: {result['formatted_prediction']}")
    print(f"Phase: {result['flight_phase']}")
    print(f"Component: {result['component_analysis']['primary_component']}")
    print(f"Confidence: {result['confidence_interval']['confidence_level']:.0%}")
    print(f"\n{result['recommendation']}")
    
    print("\n" + "="*60)
    print("Enhancement Summary:")
    print("1. Phase Normalization - Filters takeoff false positives")
    print("2. Component Mapping - AMM maintenance references")
    print("3. Confidence Intervals - Statistical uncertainty")
    print("="*60)

if __name__ == "__main__":
    quick_demo()
