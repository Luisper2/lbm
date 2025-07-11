import os
import time
from lbmpy.lbm import LBM
from lbmpy.lbm import plotter

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    nx = 256
    ny = 256

    reynolds = 100
    velocity = 0.01

    nu = velocity * nx / reynolds
    tau = 3 * nu + 0.5

    simulation = LBM(nx, ny, tau, velocity)
    simulation.run(steps = 2500000, save = 10000)

    plotter('2490000.npy', rewrite = True)

    print(f'Done ({(time.perf_counter() - start):.3f}s)')