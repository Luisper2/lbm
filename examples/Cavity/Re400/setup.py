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

    conditions = {
        'nx': nx,
        'ny': ny,
        're': reynolds,
        'tau': tau,
        'walls': ['t', 'l', 'r', 'b'],
        'periodic': [],
        'inputs': [{
            't': velocity
        }]
    }

    simulation = LBM.load_simulation('../Re200/results/npy/9990000.npy', conditions = conditions)
    simulation.run(steps = 10000000, save = 10000)

    plotter('9990000.npy', rewrite = True)

    # simulation = LBM(conditions = conditions)
    # simulation.run()
    
    # plotter('0005000.npy')

    print(f'Done ({(time.perf_counter() - start):.3f}s)')