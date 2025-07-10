from lbmpy.lbm import LBM
from lbmpy.plot import main
import time

if __name__ == '__main__':
    start = time.perf_counter()

    nx = 256
    ny = 256

    reynolds = 400
    velocity = 0.1

    nu = velocity * nx / reynolds
    tau = 3 * nu + 0.5

    sim = LBM(nx, ny, u0 = velocity, tau = tau, folder='./examples', prefix='Re200')
    sim.run(1000, 100)
    sim.run(1000, 100)

    
    # # print(tau)
    # # print(f'Done ({(time.perf_counter() - start):.3f}s)')

    # sim = LBM.from_npy(npy_file='./examples/0000900.npy', tau=tau, u0=velocity)
    # sim.run(1000, 100,'./examples', prefix='Re200')



    DATA_DIR = './examples'  # Carpeta donde están los .npy

    main(DATA_DIR)

# pip install -e .{libreria}