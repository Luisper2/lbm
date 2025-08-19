import os
import time
import jax.numpy as jnp
from lbmpy.lbm import LBM
from lbmpy.lbm import plotter

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    l = 56
    radius = 10
    cylinders = 10

    buffer = 2 * l

    top = True

    nx = 2 * buffer + 2 * radius + l * (int(cylinders / 2 + 0.5) if cylinders % 2 == 0 else int(cylinders / 2) + 1)
    ny = l

    I, J = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    mask = jnp.zeros((nx, ny), dtype=bool)

    for i in range(cylinders):
        col = i // 2 + (0.5 if i % 2 == 1 else 0)
        
        x = buffer + radius + int(col * l)
        y = int(3 * l / 4) if top else int(l / 4)

        mask = mask | ((I - x)**2 + (J - y)**2 <= radius**2)

        top = not top

    tau      = 1
    velocity = 0.1
    
    export = 'dat'

    conditions = {
        'nx': nx,
        'ny': ny,
        'tau': tau,
        'walls': [],
        'periodic': ['v'],
        'input': {
            'l': { 'u': velocity, 'v': 0 }
        }
    }

    simulation = LBM(conditions = conditions, mask = mask)
    simulation.run(steps = 10001, save = 1000, export = export)
    
    for i in range(1000, 2001, 1000):
        plotter(f'{i:07d}.{export}', rewrite = True)

    print(f'Done ({(time.perf_counter() - start):.3f}s)')