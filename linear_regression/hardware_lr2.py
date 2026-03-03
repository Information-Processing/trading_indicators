import numpy as np
import time
from pynq import allocate

#==========================================================================================================================
#Tested working for 50, 100
#==========================================================================================================================
# --- Parameters ---
D = 13
NUM_SAMPLES = 7375

# --- Register Addresses from HLS Report ---
ADDR_AP_CTRL            = 0x000
ADDR_MEM_IN_DATA        = 0x010
ADDR_NUM_SAMPLES        = 0x01c
ADDR_ATB_BASE           = 0x040 
ADDR_ATA_BASE           = 0x400 

# 2. Data Prep
test_x = np.random.uniform(0.5, 1.5, (NUM_SAMPLES, D))
test_y = np.random.uniform(0.5, 1.5, NUM_SAMPLES)
mem_in = allocate(shape=(NUM_SAMPLES, 16), dtype=np.uint32)



for s in range(NUM_SAMPLES):
    xb_raw = [int(round(x * 32)) & 0x3FFFF for x in test_x[s]] 
    yb_raw = int(round(test_y[s] * 32)) & 0x3FFFF
    
    buf = yb_raw
    for k in range(D):
        buf |= (xb_raw[k] << ((k + 1) * 18))
        
    for i in range(16):
        mem_in[s, i] = (buf >> (i * 32)) & 0xFFFFFFFF

# 3. Hardware Execution
print("Executing Hardware...")
# --- START TIMER ---
start_time = time.time()
print("Preparing Data...")

ip.write(ADDR_NUM_SAMPLES, NUM_SAMPLES)
ip.write(ADDR_MEM_IN_DATA, mem_in.device_address & 0xFFFFFFFF)
ip.write(ADDR_MEM_IN_DATA + 4, (mem_in.device_address >> 32) & 0xFFFFFFFF)



# Start and wait
ip.write(ADDR_AP_CTRL, 0x01) # Start
while not (ip.read(ADDR_AP_CTRL) & 0x02): pass # Wait for Done



# 4. Result Retrieval (AtA and Atb)
hw_ata_raw = np.zeros((D, D), dtype=np.int32)
for i in range(D*D):
    raw = ip.read(ADDR_ATA_BASE + (i * 4))
    hw_ata_raw[i // D, i % D] = raw if raw < 0x80000000 else raw - 0x100000000

hw_atb_raw = np.zeros(D, dtype=np.int32)
for i in range(D):
    raw = ip.read(ADDR_ATB_BASE + (i * 4))
    hw_atb_raw[i] = raw if raw < 0x80000000 else raw - 0x100000000

# --- STOP TIMER ---
end_time = time.time()
print("Hardware Done.")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
    
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
print(f"Execution Time: {(end_time - start_time):.6f} seconds")


print(f"--- Hardware vs Software Comparison ---")
print(f"AtA Sample (0,0) HW [Raw]: {hw_ata_raw[0,0]} | SW [Raw]: {sw_ata_final[0,0]}") 
print(f"Atb Sample (0)   HW [Raw]: {hw_atb_raw[0]}    | SW [Raw]: {sw_atb_final[0]}")                

# Final Check
print(f"AtA Match: {np.allclose(hw_ata_raw, sw_ata_final, atol=0)}")
print(f"Atb Match: {np.allclose(hw_atb_raw, sw_atb_final, atol=0)}")

# Clean up
mem_in.freebuffer()
