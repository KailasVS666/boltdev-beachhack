
import os
import sys
import subprocess
import json
import glob

def run_step(description, command):
    print(f"\n[TEST] {description}...")
    try:
        if "|" in command:
            # Handle piped input for interactive scripts (Windows friendly)
            # cmd /c "echo input | python script"
            full_cmd = f'cmd /c "{command}"'
            ret = subprocess.call(full_cmd, shell=True)
        else:
            ret = subprocess.call(command, shell=True)
            
        if ret == 0:
            print("   -> PASS")
            return True
        else:
            print(f"   -> FAIL (Exit Code: {ret})")
            return False
    except Exception as e:
        print(f"   -> ERROR: {e}")
        return False

def verify_file_exists(path, description):
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   -> PASS: {description} found ({size} bytes)")
        return True
    else:
        print(f"   -> FAIL: {description} NOT found at {path}")
        return False

def main():
    print("================================================================")
    print("   AEROGUARD IMPLEMENTATION PLAN - FINAL VERIFICATION")
    print("================================================================")
    
    all_passed = True
    
    # 1. Test Data Preparation
    # Run the script and check for JSON logs
    if run_step("Running Data Prep Module (src.data_prep_logs)", "python -m src.data_prep_logs"):
        # Check output
        logs = glob.glob("json_logs/*.json")
        if len(logs) > 0:
            print(f"   -> PASS: Found {len(logs)} JSON logs in json_logs/")
        else:
            print("   -> FAIL: No JSON logs generated.")
            all_passed = False
    else:
        all_passed = False
        
    # 2. Test Model Integration & Visualization
    # Run existing integration test
    if not run_step("Running Final Integration Test (src.final_integration_test)", "python -m src.final_integration_test"):
        all_passed = False
    
    # Check artifacts
    if not verify_file_exists("final_demo_results.png", "Integration Plot"): all_passed = False
    if not verify_file_exists("casual_flight_output.json", "Casual Flight Report"): all_passed = False
    
    # 3. Test SFI Validation & Feedback Loop
    # We use file redirection to simulate user input safely on Windows
    print("\n[TEST] Testing Interactive Feedback Loop (Simulating Rejection)...")
    
    with open("test_inputs.txt", "w") as f:
        f.write("n\nVerification Script Test Rejection\n")
        
    # python -m src.sfi_validation_ntsb < test_inputs.txt
    cmd = "python -m src.sfi_validation_ntsb < test_inputs.txt"
    
    if run_step("Running SFI Validation (Redirected Input)", cmd):
        # Verify log output
        if verify_file_exists("feedback_log.json", "Feedback Log"):
            try:
                # Give FS a moment (though subprocess block should satisfy)
                import time
                time.sleep(1) 
                
                with open("feedback_log.json", "r") as f:
                    logs = json.load(f)
                    # Check last entry
                    last_entry = logs[-1]
                    # We expect a new entry with 'accepted': False
                    if last_entry['accepted'] == False and "Verification Script" in last_entry['comments']:
                         print("   -> PASS: Rejection correctly recorded in JSON.")
                    else:
                         # Check if we just read the old one
                         print(f"   -> FAIL: Log entry mismatch. Last entry timestamp: {last_entry.get('timestamp')}")
                         print(f"      Expected comment 'Verification Script', got '{last_entry.get('comments')}'")
                         all_passed = False
            except Exception as e:
                print(f"   -> FAIL: Could not read feedback log: {e}")
                all_passed = False
    else:
        all_passed = False
        
    # Cleanup
    if os.path.exists("test_inputs.txt"):
        os.remove("test_inputs.txt")

    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("   ✅ ALL SYSTEMS GO: IMPLEMENTATION PLAN VERIFIED")
    else:
        print("   ❌ ISSUES DETECTED: CHECK LOGS ABOVE")
    print("="*60)

if __name__ == "__main__":
    main()
