# simulate_fused.py
import cupy as cp
import numpy as np
import time
from os.path import join

# -------------------------------------------------------------------
# Raw CUDA C kernel: one full Jacobi sweep per launch
# -------------------------------------------------------------------
_fused_src = r'''
extern "C" __global__
void jacobi_one(
    const float* __restrict__ u,
    float*       __restrict__ u_new,
    const unsigned char* __restrict__ mask,
    int W,
    int H
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) return;
    int idx = i * W + j;

    // Border: copy fixed boundary
    if (i == 0 || j == 0 || i == H-1 || j == W-1) {
        u_new[idx] = u[idx];
    } else {
        // neighbor indices
        int im1 = idx - W, ip1 = idx + W;
        int jm1 = idx - 1, jp1 = idx + 1;
        // interior room?
        if (mask[(i-1)*(W-2) + (j-1)]) {
            u_new[idx] = 0.25f * (
                u[im1] + u[ip1] +
                u[jm1] + u[jp1]
            );
        } else {
            // wall or outside: carry forward
            u_new[idx] = u[idx];
        }
    }
}
'''
_jacobi_one = cp.RawKernel(_fused_src, 'jacobi_one')

# -------------------------------------------------------------------
# I/O and helpers
# -------------------------------------------------------------------
def load_data(load_dir, bid):
    SIZE = 512
    # host arrays
    h_u    = np.zeros((SIZE+2, SIZE+2), dtype=np.float32)
    h_u[1:-1,1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    h_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(np.uint8)
    # transfer to device
    return cp.asarray(h_u), cp.asarray(h_mask)

def jacobi_fused(u, mask, max_iter=20000):
    H, W = u.shape
    u_new = cp.empty_like(u)
    # 2D grid of threads
    threads = (32, 16)
    blocks  = ((W + threads[0]-1)//threads[0],
               (H + threads[1]-1)//threads[1])
    # 32-bit ints for the kernel
    W32 = np.int32(W)
    H32 = np.int32(H)
    for _ in range(max_iter):
        # correct RawKernel launch syntax:
        _jacobi_one(
            blocks,      # grid dimensions
            threads,     # block dimensions
            (u, u_new, mask, W32, H32)
        )
        u, u_new = u_new, u
    return u

def summary_stats(u, mask):
    # copy only the interior block back to host
    h = u[1:-1,1:-1].get()
    m = mask.get().astype(bool)
    vals = h[m]
    return {
        'mean_temp':    vals.mean(),
        'std_temp':     vals.std(),
        'pct_above_18': (vals > 18).mean() * 100,
        'pct_below_15': (vals < 15).mean() * 100,
    }

# -------------------------------------------------------------------
# Main: write CSV to stdout
# -------------------------------------------------------------------
if __name__=='__main__':
    import sys
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR,'building_ids.txt')) as f:
        bids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv)>1 else 10
    MAX_ITER = int(sys.argv[2]) if len(sys.argv)>2 else 20000

    print('building_id,mean,std,pct>18,pct<15')
    t0 = time.time()
    for bid in bids[:N]:
        u, mask = load_data(LOAD_DIR, bid)
        u = jacobi_fused(u, mask, MAX_ITER)
        stats = summary_stats(u, mask)
        print(f"{bid},{stats['mean_temp']:.4f},"
              f"{stats['std_temp']:.4f},"
              f"{stats['pct_above_18']:.2f},"
              f"{stats['pct_below_15']:.2f}")
    # ensure all kernels finished
    cp.cuda.Stream.null.synchronize()
    total = time.time() - t0
    print(f"\nTotal wall-time for {N} builds: {total:.2f} s")
