import os
import time
from lbmpy.lbm import LBM
from lbmpy.lbm import plotter

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    nx = 256
    ny = 256

    tau      = 1
    velocity = 6e-8

    conditions = {
        'nx': nx,
        'ny': ny,
        'tau': tau,
        'walls': [],
        'periodic': ['v'],
        'inputs': [{
            'l': velocity
        }]
    }

    simulation = LBM(conditions = conditions)
    simulation.run(steps = 1001, save = 100, export = 'dat')
    
    for i in range(100, 1000, 100):
        plotter(f'{i:07d}.dat', rewrite = True)

    print(f'Done ({(time.perf_counter() - start):.3f}s)')