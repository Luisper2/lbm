import os
import time
from lbmpy.lbm import LBM
from lbmpy.lbm import plotter

if __name__ == '__main__':
    # os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    nx = 256
    ny = 256

    reynolds =100
    velocity = 0.01

    nu = velocity * nx / reynolds
    tau = 3 * nu + 0.5

    # simulation = LBM(nx, ny, tau, velocity)
    # simulation.run(1000000, 1000)

    simulation = LBM.load_simulation('./results/npy/0999000.npy')
    simulation.run(1000000, 1000)

    # plotter('./results/npy/0999000.npy', rewrite = True)
    plotter('./results/npy/1000000.npy', rewrite = True)
    plotter('./results/npy/1998000.npy', rewrite = True)

    print(f'Done ({(time.perf_counter() - start):.3f}s)')