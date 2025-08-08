import os
import jax
import copy
import matplotlib
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt

def load_data(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension == '.npy':
        file = np.load(file_path, allow_pickle=True).item()

        conditions = file.get('conditions', {})
        u          = file['u']
        v          = file['v']
        rho        = file['rho']
        mask       = file.get('mask', None)

        u   = u.T
        v   = v.T
        rho = rho.T
        if mask is not None:
            mask = mask.T

        ny, nx = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

    elif extension == '.dat':
        conditions = {}
        data_rows = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
        
                if not line or line.startswith('#'):
                    if ':' in line and not line.lower().startswith('# x y u v rho'):
                        key, val = line.lstrip('#').split(':', 1)
                        conditions[key.strip()] = val.strip()
                    continue

                parts = line.split()
        
                if len(parts) != 6:
                    continue

                data_rows.append([float(p) for p in parts])

        data = np.array(data_rows)
        i_idx     = data[:, 0].astype(int)
        j_idx     = data[:, 1].astype(int)
        u_vals    = data[:, 2]
        v_vals    = data[:, 3]
        rho_vals  = data[:, 4]
        mask_vals = data[:, 5].astype(int)

        nx = i_idx.max() + 1
        ny = j_idx.max() + 1

        u   = np.zeros((ny, nx))
        v   = np.zeros((ny, nx))
        rho = np.zeros((ny, nx))
        mask= np.zeros((ny, nx), dtype=bool)

        u[j_idx, i_idx]    = u_vals
        v[j_idx, i_idx]    = v_vals
        rho[j_idx, i_idx]  = rho_vals
        mask[j_idx, i_idx] = mask_vals == 1

        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

    else:
        raise ValueError(f'Unsupported format ({extension}). Use .npy or .dat.')

    return X, Y, u, v, rho, conditions, mask

class LBM():
    def __init__(self, conditions: dict = None, mask: jnp.ndarray = None, u: jnp.ndarray = None, v: jnp.ndarray = None, rho: jnp.ndarray = None, prefix: str = None, continue_iteration: int = 0):
        self.conditions = conditions if conditions else {
            'nx': 100,
            'ny': 100,
            'tau': 1,
            'walls': [],
            'periodic': ['v'],
            'inputs': [{
                'l': 0.1
            }]
        }

        self.nx  = self.conditions['nx']
        self.ny  = self.conditions['ny']
        self.tau = self.conditions['tau']
        
        self.dimentions = 9

        self.prefix = prefix + '_' if prefix else ''
        self.continue_iterations = continue_iteration

        self.index = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        self.wt    = jnp.array([4/9] + [1/9]*4 + [1/36]*4, dtype=jnp.float32)
        
        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=jnp.int32)
        self.ey = jnp.array([0, 0, 1,  0, -1, 1,  1, -1,-1], dtype=jnp.int32)
        
        self.bounce = jnp.array([0,3,4,1,2,7,8,5,6], dtype=jnp.int32)

        self.params = {
            't': {
                'bb':       jnp.array([2, 5, 6], dtype=jnp.int32),
                'comp':     self.ex,
                'int': {
                    'i': jnp.arange(1, self.nx+1, dtype=jnp.int32),
                    'j': self.ny
                },
                'ext':      {
                    'i': jnp.arange(1, self.nx+1, dtype=jnp.int32),
                    'j': self.ny + 1
                },
            },
            'b': {
                'bb':        jnp.array([4, 7, 8], dtype=jnp.int32),
                'comp':      self.ex,
                'int':  {
                    'i': jnp.arange(1, self.nx+1, dtype=jnp.int32),
                    'j': 1
                },
                'ext':       {
                    'i': jnp.arange(1, self.nx+1, dtype=jnp.int32),
                    'j': 0
                },
            },
            'l': {
                'bb':       jnp.array([3, 6, 7], dtype=jnp.int32),
                'comp':     self.ey,
                'int': {
                    'i': 1,
                    'j': jnp.arange(1, self.ny+1, dtype=jnp.int32)
                },
                'ext':      {
                    'i': 0,
                    'j': jnp.arange(1, self.ny+1, dtype=jnp.int32)
                },
            },
            'r': {
                'bb':       jnp.array([1, 5, 8], dtype=jnp.int32),
                'comp':     self.ey,
                'int': {
                    'i': self.nx,
                    'j': jnp.arange(1, self.ny+1, dtype=jnp.int32)
                },
                'ext':      {
                    'i': self.nx + 1,
                    'j': jnp.arange(1, self.ny+1, dtype=jnp.int32)
                },
            },
        }

        self.x, self.y = jnp.meshgrid(jnp.arange(nx+2, dtype=jnp.float32) - 0.5, jnp.arange(ny+2, dtype=jnp.float32) - 0.5, indexing='ij')

        shape = (self.nx + 2, self.ny + 2)
        
        self.u   = jnp.zeros(shape, dtype=jnp.float32)
        self.v   = jnp.zeros(shape, dtype=jnp.float32)
        self.rho = jnp.ones(shape,  dtype=jnp.float32)
        
        if u is not None:
            self.u = u
        if v is not None:
            self.v = v
        if rho is not None:
            self.rho = rho

        self.f = self.equilibrium(self.u, self.v, self.rho)
        self.ferr = jnp.zeros_like(self.f)

        expected = (self.nx, self.ny)

        if mask is not None and mask.shape == expected:
            self.mask = jnp.pad(mask, pad_width=((1,1), (1,1)), constant_values=False)
        else:
            self.mask = jnp.zeros((self.nx+2, self.ny+2), dtype=bool)


        for wall in self.conditions.get('walls', []):
            ext = self.params[wall]['ext']

            self.mask = self.mask.at[(ext['i'], ext['j'])].set(True)

    @partial(jax.jit, static_argnums=0)
    def equilibrium(self, u: jnp.ndarray = None, v: jnp.ndarray = None, rho: jnp.ndarray = None) -> jnp.ndarray:
        usq = u**2 + v**2
        cu  = (u[None] * self.ex[:,None,None] +
               v[None] * self.ey[:,None,None])
        feq = rho[None] * self.wt[:,None,None] * (
              1 + 3*cu + 4.5*cu**2 - 1.5*usq[None]
        )

        return feq

    @partial(jax.jit, static_argnums=0)
    def collide(self, f: jnp.ndarray = None, feq: jnp.ndarray = None)  -> jnp.ndarray:
        return f - (f - feq) / self.tau
        
    @partial(jax.jit, static_argnums=0)
    def stream(self, f: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        def shift_fk(fk, dx, dy):
            return jnp.roll(jnp.roll(fk, dx, axis=0), dy, axis=1)

        ftemp   = f
        f_steam = jnp.stack([shift_fk(f[k], self.ex[k], self.ey[k]) for k in range(self.dimentions)])

        return f_steam, ftemp

    @partial(jax.jit, static_argnums=0)
    def bounce_back(self, f: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([jnp.where(self.mask, f[self.bounce[k]], f[k]) for k in range(self.dimentions)], axis=0)

    @partial(jax.jit, static_argnums=0)
    def neumann_bc(self, f: jnp.ndarray = None, ftemp: jnp.ndarray = None, rho: jnp.ndarray = None) -> jnp.ndarray:
        walls  = self.conditions.get('walls', [])
        inputs = self.conditions.get('inputs', [])

        u_vals = {
            side: inp[side]
            for inp in inputs
            for side in ('t','b','l','r')
            if side in inp
        }

        for side in walls:
            if side not in self.params or side not in u_vals:
                continue

            p      = self.params[side]
            k_idx  = p['bb']
            comp   = p['comp']
            u_wall = u_vals[side]

            if side in ('t','b'):
                i_idx = p['int']['i']
                j     = p['int']['j']
                K, I  = jnp.meshgrid(k_idx, i_idx, indexing='ij')
                J     = jnp.full_like(I, j)

            else:
                j_idx = p['int']['j']
                i     = p['int']['i']
                K, J  = jnp.meshgrid(k_idx, j_idx, indexing='ij')
                I     = jnp.full_like(J, i)

            term     = 6 * self.wt[K] * rho[I, J] * comp[K] * u_wall
            new_vals = ftemp[K, I, J] - term
            bb       = self.bounce[K]

            f = f.at[bb, I, J].set(new_vals)

        return f

    @partial(jax.jit, static_argnums=0)
    def macroscopic(self, f: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rho = jnp.sum(f, axis=0)
        nonzero = rho > 1e-8

        u = jnp.sum(f * self.ex[:,None,None], axis=0) / jnp.where(nonzero, rho, 1.0)
        v = jnp.sum(f * self.ey[:,None,None], axis=0) / jnp.where(nonzero, rho, 1.0)

        u = jnp.where(nonzero, u, 0.0)
        v = jnp.where(nonzero, v, 0.0)

        return u, v, rho

    @partial(jax.jit, static_argnums=0)
    def step(self, f: jnp.ndarray = None, u: jnp.ndarray = None, v: jnp.ndarray = None, rho: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        feq         = self.equilibrium(u, v, rho)
        fcol        = self.collide(f, feq)
        fstr, ftemp = self.stream(fcol)
        fbb         = self.bounce_back(fstr)
        fneu        = self.neumann_bc(fbb, ftemp, rho)
        u, v, rho   = self.macroscopic(fneu)
        
        return fneu, u, v, rho

    def save(self, dir: str, export: str = 'npy', iteration: int = 0, u: jnp.ndarray = None, v: jnp.ndarray = None, rho: jnp.ndarray = None, conditions: dict = {}, mask: jnp.ndarray = None) -> None:
        name = os.path.join(dir, f'{self.prefix}{iteration:07d}.{export}')

        if export == 'npy':
            jnp.save(name, { 'u': jnp.array(u), 'v': jnp.array(v), 'rho': jnp.array(rho), 'conditions': conditions, 'mask': mask })
        elif export == 'dat':
            u_np   = np.asarray(u)
            v_np   = np.asarray(v)
            rho_np = np.asarray(rho)
            mask_np= np.asarray(mask)
            nx, ny = u_np.shape

            with open(name, 'w') as f:
                f.write('# X Y U V RHO MASK\n')
                
                for key, val in conditions.items():
                    f.write(f'# {key}: {val}\n')

                for i in range(nx):
                    for j in range(ny):
                        m = int(mask_np[i, j])
                        f.write(
                            f'{i} {j} '
                            f'{u_np[i, j]:.6e} '
                            f'{v_np[i, j]:.6e} '
                            f'{rho_np[i, j]:.6e} '
                            f'{m}\n'
                        )
        else:
            raise ValueError(f'Unsupported format ({export}). Use .npy or .dat.')
    
    def run(self, steps: int = 5001, save: int = 100, dir = os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), 'results'), export: str = 'npy'):
        save_path = os.path.join(dir, export)
        os.makedirs(save_path, exist_ok = True)

        f, mask, u, v, rho = self.f, self.mask, self.u, self.v, self.rho
        
        for it in tqdm(range(self.continue_iterations, self.continue_iterations + steps)):
            f, u, v, rho = self.step(f, u, v, rho)

            if it % save == 0:
                self.conditions['iteration'] = it

                self.save(save_path, export, it, u, v, rho, self.conditions, mask)

        self.f, self.u, self.v, self.rho = f, u, v, rho

    @classmethod
    def load_simulation(cls, file_dir: str = '0000000.npy', conditions: dict = None, mask: jnp.ndarray = None, prefix: str = None, continue_iteration: bool = True):
        X, Y, u, v, rho, load_conditions, load_mask = load_data(os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), file_dir))

        if conditions is not None:
            load_conditions.update(conditions)

        if mask is not None:
            load_mask = mask

        return cls(conditions = load_conditions, mask = load_mask, u = u.T, v = v.T, rho = rho.T, prefix = prefix, continue_iteration = int(load_conditions['iteration']) if continue_iteration else 0)

def plotter (dir: str = '0000000.npy', save_dir = None, rewrite: bool = False, velocity: bool = True, density: bool = True, vorticity: bool = True):
    def create_plot(file_path, X, Y, data, u = None, v = None, title: str = '', label: str = ''):
        width = X.max() - X.min()
        height = Y.max() - Y.min()
        aspect_ratio = width / height

        base_height = 6
        figsize = (aspect_ratio * base_height, base_height)

        fig, ax = plt.subplots(figsize=figsize)

        mesh = ax.pcolormesh(X, Y, data, shading='auto', cmap='rainbow')

        fig.colorbar(mesh, ax=ax, label=label)

        if u is not None and v is not None:
            ax.streamplot(X, Y, u, v, density=1, color='k', linewidth=0.7, arrowsize=0.5)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        ax.set_xlim(X.min()+1, X.max()-1)
        ax.set_ylim(Y.min()+1, Y.max()-1)

        ax.set_aspect('auto')

        fig.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    matplotlib.use('Agg')

    extension = Path(dir).suffix.lstrip('.')
    path = os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), 'results', extension, f'{dir}')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dir/File not exsist: {path}')
    
    save = os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), 'results', save_dir) if save_dir else os.path.join(os.path.dirname(os.path.abspath(__import__('__main__').__file__)), 'results', 'plots')
    
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.endswith(f'.{extension}')
    ] if os.path.isdir(path) else [path] if path.endswith(f'.{extension}') else []

    if len(files) == 0:
        raise FileNotFoundError(f'Not .{extension} file(s) founded')
    
    os.makedirs(save, exist_ok = True)

    for file in files:
        base = Path(os.path.basename(file)).stem
        
        cache_path = os.path.join(save, base)

        os.makedirs(cache_path, exist_ok = True)

        X, Y, u, v, rho, conditions, mask = load_data(file)

        X = X[1:-1, 1:-1]
        Y = Y[1:-1, 1:-1]
        u = u[1:-1, 1:-1]
        v = v[1:-1, 1:-1]
        rho = rho[1:-1, 1:-1]

        remove = ['nx', 'ny', 'walls', 'periodic', 'inputs']  
        cache = copy.deepcopy(conditions)

        for key in remove:
            cache.pop(key, None)

        remove = ['nx', 'ny', 'walls', 'periodic', 'inputs']
        cache = conditions.copy()

        for key in remove:
            cache.pop(key, None)

        name = f'{', '.join([*[f'{k}={cache[k]}' for k in cache]])} ({conditions['nx']}, {conditions['ny']})' or base

        velocity_path = os.path.join(cache_path, 'Velocity.png')
        voritcity_path = os.path.join(cache_path, 'Vorticity.png')
        density_path = os.path.join(cache_path, 'Density.png')

        if velocity and (rewrite or not os.path.exists(velocity_path)):
            data = np.sqrt(u ** 2 + v ** 2)
            create_plot(velocity_path, X, Y, data, u, v, name, 'Velocity')

        if vorticity and (rewrite or not os.path.exists(voritcity_path)):
            data = np.gradient(v, axis = 1) - np.gradient(u, axis = 0)
            create_plot(voritcity_path, X, Y, data, None, None, name, 'Vorticity')
        
        if density and (rewrite or not os.path.exists(density_path)):
            create_plot(density_path, X, Y, rho, None, None, name, 'Density')