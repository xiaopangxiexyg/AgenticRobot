import time
import numpy as np
from inspire_tac_modbus import InspireTacModbus, InspireSimulatedForceControl

def measure_frequency(func, duration=3.0, name="Function"):
    """
    Measure the calling frequency of a function.
    
    Args:
        func: The function to execute.
        duration: How long to run the test in seconds.
        name: Name of the function for display.
    """
    print(f"\nTesting {name} frequency for {duration} seconds...")
    count = 0
    start_time = time.time()
    
    # Run the loop
    while time.time() - start_time < duration:
        func()
        count += 1
        
    end_time = time.time()
    elapsed = end_time - start_time
    freq = count / elapsed
    print(f"-> Result: {count} calls in {elapsed:.4f}s => Average Frequency: {freq:.2f} Hz")
    return freq

def main():
    print("Initializing InspireTacModbus interface...")
    try:
        # Instantiate the interface (using default IP from the class definition)
        hand = InspireTacModbus()
        
        print("Connected successfully.")
        time.sleep(1)

        # 1. Test Reading Joint Positions
        # This reads 6 registers
        measure_frequency(lambda: hand.read_joint_pos(), name="read_joint_pos (Read 6 Regs)")

        # 2. Test Reading Tactile Data
        # This reads a large amount of registers (approx 1062 data points)
        measure_frequency(lambda: hand.read_tac_data(), name="read_tac_data (Read Tac Block)")

        # 2.1 Test Partial Reading Tactile Data (Fingertips)
        fingertips = ["little_1", "ring_1", "middle_1", "index_1", "thumb_1"]
        measure_frequency(lambda: hand.read_tac_data(target_regions=fingertips), name="read_tac_data (Partial: Fingertips)")
        
        # Verify data correctness for partial read
        print("\nVerifying partial read correctness...")
        vec, data = hand.read_tac_data(target_regions=fingertips)
        print("Keys in partial read dict:", list(data.keys()))
        if set(data.keys()) == set(fingertips):
             print("Verified: Returned keys match requested regions.")
        else:
             print(f"Error: Expected keys {fingertips}, got {list(data.keys())}")

        # Check unrequested region (palm)
        if 'palm' in hand.tac_region_names:
            i_palm = hand.tac_region_names.index('palm')
            start_palm = hand.tac_region_start_idxs[i_palm]
            end_palm = hand.tac_region_start_idxs[i_palm+1]
            
            # In partial read, palm should be 0
            palm_vals = vec[start_palm:end_palm]
            if np.all(palm_vals == 0):
                print("Verified: Unrequested region 'palm' is zero in partial read.")
            else:
                print("Warning: Unrequested region 'palm' has non-zero data (might be overlapping chunk or noise).")

        # Preparation for Control Tests
        # Read current position first to ensure we verify safety and don't move the hand unintentionally
        current_pos = hand.read_joint_pos()
        print(f"\nCurrent joint positions: {np.round(current_pos, 3)}")
        print("Using current position for 'set_joint_pos' test to avoid movement.")
        
        # 3. Test Setting Joint Positions
        # This writes 6 registers
        measure_frequency(lambda: hand.set_joint_pos(current_pos), name="set_joint_pos (Write 6 Regs)")

        # 4. Test Setting Force
        # This writes 6 registers
        # Using a safe force value (e.g. 800 which is default in init)
        default_force = [800] * 6
        measure_frequency(lambda: hand.set_force(default_force), name="set_force (Write 6 Regs)")



        controller = InspireSimulatedForceControl()
    
        try:
            # 示例: 逐渐握紧
            print("Start Force Control Demo...")
            
            # 1. 先完全张开
            print("Opening...")
            controller.step([1.0]*6, [0.5]*6) # 目标位置1(开), 力设置无所谓(内部会覆盖为Max)
            time.sleep(2)
            
            # 2. 模拟柔顺抓取 (Soft Grasp)
            # 设定目标位置为 0.0 (完全闭合)
            # 设定力矩比例为 0.2 (20%的最大力)
            print("Soft Grasping...")
            start_time = time.time()
            while time.time() - start_time < 3.0:
                controller.get_state() # 更新状态
                controller.step(target_pos=[0.0]*6, target_close_force_ratio=[0.2]*6)
                time.sleep(0.02)
            
            time.sleep(3)
                
            # 3. 模拟用力抓取 (Hard Grasp)
            # 保持位置目标 0.0
            # 增大目标力到 0.8
            print("Hard Grasping...")
            start_time = time.time()
            while time.time() - start_time < 3.0:
                controller.get_state()
                controller.step(target_pos=[0.0]*6, target_close_force_ratio=[0.8]*6)
                time.sleep(0.02)
                
            # 4. 释放
            print("Releasing...")
            controller.step(target_pos=[1.0]*6, target_close_force_ratio=[0.5]*6)
            time.sleep(2)
            
        finally:
            controller.close()


        print("\nClosing connection...")
        hand.close()
        print("Done.")

    except Exception as e:
        print(f"\n[Error] An exception occurred during testing: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
