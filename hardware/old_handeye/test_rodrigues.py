import cv2
import numpy as np

def test_rodrigues_stability():
    # Axis of rotation: Z axis
    axis = np.array([0, 0, 1], dtype=np.float32)
    
    # Angles close to 180 degrees (pi)
    angles = [np.pi - 0.01, np.pi, np.pi + 0.01]
    
    print("Testing Rodrigues stability around 180 degrees...")
    
    for angle in angles:
        # Create rotation vector
        rvec_in = axis * angle
        # Convert to matrix
        R, _ = cv2.Rodrigues(rvec_in)
        # Convert back to vector
        rvec_out, _ = cv2.Rodrigues(R)
        
        print(f"Angle: {angle:.4f}")
        print(f"Input rvec:  {rvec_in}")
        print(f"Output rvec: {rvec_out.flatten()}")
        
        # Check if matrix reconstructed from output rvec is same as R
        R_recon, _ = cv2.Rodrigues(rvec_out)
        diff = np.linalg.norm(R - R_recon)
        print(f"Matrix diff: {diff:.6f}")
        print("-" * 20)

if __name__ == "__main__":
    test_rodrigues_stability()
