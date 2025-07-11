import os
import time
from lbmpy.lbm import LBM
from lbmpy.lbm import plotter

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    nx = 256
    ny = 256

    reynolds = 400
    velocity = 0.1

    nu = velocity * nx / reynolds
    tau = 3 * nu + 0.5

    simulation = LBM(nx, ny, tau, velocity)
    simulation.run(steps = 10000, save = 100)

    simulation = LBM.load_simulation('0009000.npy')
    simulation.run(steps = 10000, save = 100)

    plotter('0009000.npy', rewrite = True)
    plotter('0010000.npy', rewrite = True)
    plotter('0018000.npy', rewrite = True)

    print(f'Done ({(time.perf_counter() - start):.3f}s)')