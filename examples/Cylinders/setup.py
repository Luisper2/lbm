import os
os.environ["JAX_PLATFORMS"] = "gpu"
import time
import jax.numpy as jnp
from lbmpy.lbm import LBM

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    
    start = time.perf_counter()

    l = 56
    radius = 10
    cylinders = 10

    buffer = 2 * l

    max_col = (cylinders - 1) // 2 + (0.5 if (cylinders % 2 == 0 and cylinders > 0) else 0.0)

    nx = int(2 * buffer + 2 * radius + l * max_col)
    ny = l

    I, J = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing='ij')
    mask = jnp.zeros((nx, ny), dtype=bool)

    top = True
    for i in range(cylinders):
        col = i // 2 + (0.5 if i % 2 == 1 else 0.0)
        
        x = buffer + radius + col * l
        y = 3 * l / 4 if top else l / 4
        
        mask = mask | ((I - x)**2 + (J - y)**2 <= radius**2)
        
        top = not top

    tau      = 1
    velocity = 0.01
    
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
    simulation.run(steps = 10000001, save = 1000000, export = export, plotting = True)
    
    print(f'Done ({(time.perf_counter() - start):.3f}s)')