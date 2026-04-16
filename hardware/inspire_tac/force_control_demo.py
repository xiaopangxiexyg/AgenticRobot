import time
import numpy as np
import sys
import os

# Ensure we can import the driver from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inspire_tac_modbus import InspireSimulatedForceControl

def main():
    print("Inspire Hand Force Control Demo")
    print("-------------------------------")
    print("Demonstrating Impedance Control: Force increases as the hand encounters resistance while closing.")
    print("1.0 = Fully Open, 0.0 = Fully Closed")
    
    # Configuration
    HAND_IP = '192.168.11.210' # Default IP
    KP = 2000                  # Stiffness (Gain for Position Error -> Force)
    MIN_FORCE = 300            # Minimum force to ensure movement
    MAX_FORCE = 800            # Max force limit for safety in demo
    
    # Control Parameters for Speed
    # Duration in seconds for the opening/closing motions
    # Increase these values to slow down the execution
    OPEN_DURATION = 0.5        
    CLOSE_DURATION = 8.0       
    
    print(f"Connecting to {HAND_IP}...")
    
    try:
        # Initialize the impedance controller wrapper
        # max_fingertip_force in InspireTacModbus init is passed from max_force_rating here?
        # Let's check init: InspireSimulatedForceControl(ip, max_force_rating...) calls InspireTacModbus(ip, max_fingertip_force=max_force_rating)
        controller = InspireSimulatedForceControl(
            ip=HAND_IP, 
            max_force_rating=MAX_FORCE, 
            kp=KP, 
            min_force_threshold=MIN_FORCE
        )
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("Connection successful.")
    
    try:
        # Phase 1: Open Hand
        print("\n[Phase 1] Opening Hand...")
        target_open = np.ones(6) * 1.0
        
        # Execute for 2 seconds to ensure open
        start_t = time.time()
        while time.time() - start_t < 2.0:
            controller.step(target_open)
            time.sleep(0.01)
            
        print("Hand Opened. Starting Cyclic Grasp Test.")
        print("Press Ctrl+C to exit.")
        
        def move_smoothly(target_pos, duration, label):
            print(f"Action: {label} (Duration: {duration:.1f}s)")
            start_time = time.time()
            start_pos = np.array(controller.get_state())
            
            while True:
                elapsed = time.time() - start_time
                if elapsed > duration:
                    break
                
                # Calculate interpolation factor (0.0 to 1.0)
                alpha = elapsed / duration
                
                # Linear Interpolation: Current Target = Start + (End - Start) * alpha
                current_target_cmd = start_pos + (target_pos - start_pos) * alpha
                
                # Step the controller with the INTERPOLATED target
                # This ensures the "virtual spring" reference moves slowly
                controller.step(current_target_cmd)
                
                # Get feedback for display
                real_pos = controller.get_state()
                avg_pos = np.mean(real_pos)
                
                # Print status
                print(f"\rProgress: {alpha*100:5.1f}% | Avg Pos: {avg_pos:.3f} | Cmd: {np.mean(current_target_cmd):.3f}", end="")
                
                time.sleep(0.005) # control loop rate approx 50Hz for this demo script
            
            print(" -> Done.")
            # Ensure final target is applied
            controller.step(target_pos)

        cycle_count = 0
        
        while True:
            cycle_count += 1
            print(f"\n--- Cycle {cycle_count} ---")
            
            # 1. Open
            # print("Action: Opening (Low Force)")
            move_smoothly(target_open, OPEN_DURATION, "Opening")
            time.sleep(1.0) # Pause between moves
                
            # 2. Impedance Grasp (Closing)
            # print("Action: Grasping (Impedance Control)")
            # print("Try blocking a finger to feel the resistance force increase.")
            
            # Target is fully closed (0.0)
            target_close = np.zeros(6)
            target_close[4] = 1
            target_close[5] = 1
            
            move_smoothly(target_close, CLOSE_DURATION, "Grasping (Impedance)")
            
            print("Holding grasp for 2 seconds...")
            start_hold = time.time()
            while time.time() - start_hold < 2.0:
                 controller.step(target_close)
                 time.sleep(0.05)


    except KeyboardInterrupt:
        print("\nStopped by user.")
        
    except Exception as e:
        print(f"\nRuntime Error: {e}")
        
    finally:
        print("Releasing resources...")
        try:
            controller.close()
        except:
            pass

if __name__ == "__main__":
    main()
