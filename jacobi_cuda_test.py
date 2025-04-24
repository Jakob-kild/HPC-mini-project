import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from os.path import join
import sys
import time

# Load simulation data
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# CUDA kernel: performs a single Jacobi iteration
@cuda.jit
def jacobi_cuda_kernel(u, u_new, mask):
    i, j = cuda.grid(2)
    if i < u.shape[0] and j < u.shape[1]:
        # border: carry forward fixed boundary
        if i == 0 or j == 0 or i == u.shape[0]-1 or j == u.shape[1]-1:
            u_new[i, j] = u[i, j]
        else:
            if mask[i-1, j-1]:
                # interior room
                u_new[i, j] = 0.25 * (
                    u[i-1, j] + u[i+1, j]
                  + u[i, j-1] + u[i, j+1]
                )
            else:
                # wall or outside
                u_new[i, j] = u[i, j]

def jacobi_cuda(u0, interior_mask, max_iter=20000):
    # copy once
    u_d      = cuda.to_device(u0)
    u_new_d  = cuda.device_array_like(u0)
    mask_d   = cuda.to_device(interior_mask.astype(np.uint8))

    # tuned block/grid
    threads_per_block = (8, 32)
    blocks_per_grid = (
        (u0.shape[0] + threads_per_block[0] - 1)//threads_per_block[0],
        (u0.shape[1] + threads_per_block[1] - 1)//threads_per_block[1],
    )

    for _ in range(max_iter):
        jacobi_cuda_kernel[blocks_per_grid, threads_per_block](u_d, u_new_d, mask_d)
        u_d, u_new_d = u_new_d, u_d

    # ensure completion before copy
    cuda.synchronize()
    return u_d.copy_to_host()

# Compute summary statistics
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }

# Optional: visualize the output
def visualize(u, building_id):
    plt.imshow(u, cmap="hot")
    plt.title(f"Simulated Temperature - Building {building_id}")
    plt.colorbar(label="Temperature (°C)")
    plt.savefig(f"plots/{building_id}_cuda_result.png")
    plt.close()

# Main routine
if __name__ == "__main__":
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 10
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]
    MAX_ITER = 20000

    print("building_id,mean_temp,std_temp,pct_above_18,pct_below_15")

    start = time.time()
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        u_final = jacobi_cuda(u0, interior_mask, MAX_ITER)
        stats = summary_stats(u_final, interior_mask)
        print(f"{bid},{stats['mean_temp']},{stats['std_temp']},{stats['pct_above_18']},{stats['pct_below_15']}")
        # visualize(u_final, bid)
    end = time.time()

    print(f"\nTotal time for {N} floorplans: {end - start:.2f} seconds")
