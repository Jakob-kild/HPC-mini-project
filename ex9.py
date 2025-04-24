# simulate_cupy.py
import cupy as cp
import numpy as np
import time
from os.path import join

def load_data(load_dir, bid):
    # host loads with NumPy, then we transfer to CuPy
    SIZE = 512
    h_u = np.zeros((SIZE+2, SIZE+2), dtype=np.float32)
    h_u[1:-1,1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    h_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool)
    # transfer to GPU
    u       = cp.asarray(h_u)
    interior= cp.asarray(h_mask)
    return u, interior

def jacobi_cupy(u, interior, max_iter=20000):
    u_new = cp.empty_like(u)
    for _ in range(max_iter):
        # slice into the *current* arrays each iteration
        u_mid  = u[1:-1, 1:-1]
        un_mid = u_new[1:-1, 1:-1]
        mask   = interior

        # copy boundaries
        u_new[0, :]  = u[0, :]
        u_new[-1, :] = u[-1, :]
        u_new[:, 0]  = u[:, 0]
        u_new[:, -1] = u[:, -1]

        # 4-point stencil
        stencil = 0.25 * (
            u[:-2, 1:-1] + u[2:,  1:-1]
          + u[1:-1, :-2] + u[1:-1, 2:]
        )

        # fused update: interior rooms←stencil, walls←old u_mid
        un_mid[:] = cp.where(mask, stencil, u_mid)

        # swap buffers
        u, u_new = u_new, u

    return u


def summary_stats(u, interior):
    # bring back the result to host
    h_u = cp.asnumpy(u)
    h_mask = cp.asnumpy(interior)
    interior_vals = h_u[1:-1,1:-1][h_mask]
    return {
        'mean_temp':  interior_vals.mean(),
        'std_temp':   interior_vals.std(),
        'pct_above_18': (interior_vals>18).mean()*100,
        'pct_below_15': (interior_vals<15).mean()*100,
    }

if __name__=='__main__':
    import sys
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR,'building_ids.txt')) as f:
        bids = f.read().splitlines()
    N = int(sys.argv[1]) if len(sys.argv)>1 else 10
    MAX_ITER = int(sys.argv[2]) if len(sys.argv)>2 else 20000

    print('building_id,mean,std,pct>18,pct<15')
    start = time.time()
    for bid in bids[:N]:
        u,mask = load_data(LOAD_DIR, bid)
        u = jacobi_cupy(u, mask, MAX_ITER)
        stats = summary_stats(u, mask)
        print(f"{bid},{stats['mean_temp']:.4f},{stats['std_temp']:.4f},"
              f"{stats['pct_above_18']:.2f},{stats['pct_below_15']:.2f}")
    cp.cuda.Stream.null.synchronize()
    total = time.time()-start
    print(f"\nTotal wall-time for {N} builds: {total:.2f} s")
