import numpy as np
import time
#==========================================================================================================================
#Tested working for 50, 100
#==========================================================================================================================
# --- Parameters ---
D = 13
NUM_SAMPLES = 7375


# 2. Data Prep
test_x = np.random.uniform(0.5, 1.5, (NUM_SAMPLES, D))
test_y = np.random.uniform(0.5, 1.5, NUM_SAMPLES)

print("Preparing Data...")
# 5. Software Simulation
print("Simulating in Software...")
# --- START TIMER ---
start_time = time.time()
test_x_quantized = np.round(test_x * 32).astype(np.int64)
test_y_quantized = np.round(test_y * 32).astype(np.int64)

sw_ata_acc = test_x_quantized.T @ test_x_quantized
sw_atb_acc = test_x_quantized.T @ test_y_quantized

# Based on your finding that '>> 9' works for AtA
sw_ata_final = sw_ata_acc >> 5
sw_atb_final = sw_atb_acc >> 5


end_time = time.time()
print("Software Done.")
print(sw_ata_final)
print(f"Execution Time: {(end_time - start_time):.6f} seconds")

