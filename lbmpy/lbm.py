import ast
import copy
import json
import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm

matplotlib.use("Agg")


def load_data(file_path):
    def _coerce(val: str):
        v = val.strip()
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(v)
        except Exception:
            return v

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".npy":
        file = np.load(file_path, allow_pickle=True).item()

        conditions = file.get("conditions", {})
        u = file["u"].T
        v = file["v"].T
        rho = file["rho"].T
        pressure = file["pressure"].T
        mask = file.get("mask", None)
        if mask is not None:
            mask = mask.T

        ny, nx = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
    elif extension == ".dat":
        conditions = {}
        data_rows = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    if ":" in line and not line.lower().startswith(
                        "# x y u v rho pressure"
                    ):
                        key, val = line.lstrip("#").split(":", 1)
                        conditions[key.strip()] = _coerce(val)
                    continue

                parts = line.split()

                if len(parts) not in (6, 7):
                    continue

                data_rows.append([float(p) for p in parts])

        if not data_rows:
            raise ValueError(
                f"No se encontraron datos válidos en {file_path}. "
                f"¿Las filas tienen 6 o 7 columnas?"
            )

        data = np.array(data_rows)
        i_idx = data[:, 0].astype(int)
        j_idx = data[:, 1].astype(int)
        u_vals = data[:, 2]
        v_vals = data[:, 3]
        rho_vals = data[:, 4]
        pressure_vals = data[:, 5]

        nx = i_idx.max() + 1
        ny = j_idx.max() + 1

        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        rho = np.zeros((ny, nx))
        pressure = np.zeros((ny, nx))

        u[j_idx, i_idx] = u_vals
        v[j_idx, i_idx] = v_vals
        rho[j_idx, i_idx] = rho_vals
        pressure[j_idx, i_idx] = pressure_vals

        mask = None
        if data.shape[1] == 7:
            mask_vals = data[:, 6].astype(int)
            mask = np.zeros((ny, nx), dtype=bool)
            mask[j_idx, i_idx] = mask_vals == 1

        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
    else:
        raise ValueError(f"Unsupported format ({extension}). Use .npy or .dat.")

    return X, Y, u, v, rho, pressure, conditions, mask


class LBM:
    def __init__(
        self,
        conditions: dict = {},
        mask: jnp.ndarray = None,
        u: jnp.ndarray = None,
        v: jnp.ndarray = None,
        rho: jnp.ndarray = None,
        pressure: jnp.ndarray = None,
        prefix: str = None,
        continue_iteration: int = 0,
    ):

        self.conditions = copy.deepcopy(conditions)
        self.nx = self.conditions["nx"]
        self.ny = self.conditions["ny"]
        self.tau = self.conditions["tau"]
        self.body_force = self.conditions.get("body_force", (0.0, 0.0))

        self.dimentions = 9

        self.c = jnp.float32(1.0 / 3.0)

        self.prefix = prefix + "_" if prefix else ""
        self.continue_iterations = continue_iteration

        self.index = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=jnp.int32)
        self.wt = jnp.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=jnp.float32)

        self.ex = jnp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=jnp.int32)
        self.ey = jnp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=jnp.int32)

        self.bounce = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)

        self.params = {
            "t": {
                "bb": jnp.array([2, 5, 6], dtype=jnp.int32),
                "comp": self.ex,
                "int": {"i": jnp.arange(1, self.nx + 1, dtype=jnp.int32), "j": self.ny},
                "ext": {
                    "i": jnp.arange(1, self.nx + 1, dtype=jnp.int32),
                    "j": self.ny + 1,
                },
            },
            "b": {
                "bb": jnp.array([4, 7, 8], dtype=jnp.int32),
                "comp": self.ex,
                "int": {"i": jnp.arange(1, self.nx + 1, dtype=jnp.int32), "j": 1},
                "ext": {"i": jnp.arange(1, self.nx + 1, dtype=jnp.int32), "j": 0},
            },
            "l": {
                "bb": jnp.array([3, 6, 7], dtype=jnp.int32),
                "comp": self.ey,
                "int": {"i": 1, "j": jnp.arange(1, self.ny + 1, dtype=jnp.int32)},
                "ext": {"i": 0, "j": jnp.arange(1, self.ny + 1, dtype=jnp.int32)},
            },
            "r": {
                "bb": jnp.array([1, 5, 8], dtype=jnp.int32),
                "comp": self.ey,
                "int": {"i": self.nx, "j": jnp.arange(1, self.ny + 1, dtype=jnp.int32)},
                "ext": {
                    "i": self.nx + 1,
                    "j": jnp.arange(1, self.ny + 1, dtype=jnp.int32),
                },
            },
            "h": [0, 2, 4],
            "v": [0, 1, 3],
        }

        self.x, self.y = jnp.meshgrid(
            jnp.arange(self.nx + 2, dtype=jnp.float32) - 0.5,
            jnp.arange(self.ny + 2, dtype=jnp.float32) - 0.5,
            indexing="ij",
        )

        shape = (self.nx + 2, self.ny + 2)

        self.u = jnp.zeros(shape, dtype=jnp.float32)
        self.v = jnp.zeros(shape, dtype=jnp.float32)
        self.rho = jnp.ones(shape, dtype=jnp.float32)
        self.p = jnp.ones(shape, dtype=jnp.float32)

        if u is not None:
            self.u = u
        if v is not None:
            self.v = v
        if rho is not None:
            self.rho = rho

        expected = (self.nx, self.ny)

        self.f = self.equilibrium(self.u, self.v, self.rho)
        self.ferr = jnp.zeros_like(self.f)

        if mask is not None and mask.shape == expected:
            self.mask = jnp.pad(mask, pad_width=((1, 1), (1, 1)), constant_values=False)
        elif mask is not None and mask.shape == shape:
            self.mask = mask
        else:
            self.mask = jnp.zeros((self.nx + 2, self.ny + 2), dtype=bool)

        for wall in self.conditions.get("walls", []):
            ext = self.params[wall]["ext"]

            self.mask = self.mask.at[(ext["i"], ext["j"])].set(True)

    @partial(jax.jit, static_argnums=0)
    def equilibrium(
        self, u: jnp.ndarray = None, v: jnp.ndarray = None, rho: jnp.ndarray = None
    ) -> jnp.ndarray:
        u = u + self.tau / rho * self.body_force[0]
        v = v + self.tau / rho * self.body_force[1]

        usq = u**2 + v**2
        cu = u[None] * self.ex[:, None, None] + v[None] * self.ey[:, None, None]
        feq = (
            rho[None]
            * self.wt[:, None, None]
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq[None])
        )

        return feq

    @partial(jax.jit, static_argnums=0)
    def collide(self, f: jnp.ndarray = None, feq: jnp.ndarray = None) -> jnp.ndarray:
        return f - (f - feq) / self.tau

    @partial(jax.jit, static_argnums=0)
    def stream(self, f: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        def shift_fk(fk, dx, dy):
            return jnp.roll(jnp.roll(fk, dx, axis=0), dy, axis=1)

        ftemp = f
        shifts = [(self.ex[k], self.ey[k]) for k in range(self.dimentions)]
        f_stream = jnp.stack(
            [
                jnp.roll(jnp.roll(f[k], dx, axis=0), dy, axis=1)
                for k, (dx, dy) in enumerate(shifts)
            ]
        )

        return f_stream, ftemp

    @partial(jax.jit, static_argnums=0)
    def periodic(self, f: jnp.ndarray) -> jnp.ndarray:
        periodic = self.conditions.get("periodic", [])

        if "h" in periodic or "horizontal" in periodic:
            f = f.at[:, 0, :].set(f[:, -2, :])
            f = f.at[:, -1, :].set(f[:, 1, :])

        if "v" in periodic or "vertical" in periodic:
            f = f.at[:, :, 0].set(f[:, :, -2])
            f = f.at[:, :, -1].set(f[:, :, 1])

        return f

    @partial(jax.jit, static_argnums=0)
    def bounce_back(self, f: jnp.ndarray, ftemp: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(
            [
                jnp.where(self.mask, ftemp[self.bounce[k]], f[k])
                for k in range(self.dimentions)
            ],
            axis=0,
        )

    @partial(jax.jit, static_argnums=0)
    def neumann_bc(
        self, f: jnp.ndarray = None, ftemp: jnp.ndarray = None, rho: jnp.ndarray = None
    ) -> jnp.ndarray:
        inp_dict = {}

        single = self.conditions.get("input", None)
        if isinstance(single, dict):
            for side in ("t", "b", "l", "r"):
                if side in single:
                    inp_dict[side] = single[side]

        multi = self.conditions.get("input", None)

        if isinstance(multi, (list, tuple)):
            for inp in multi:
                for side in ("t", "b", "l", "r"):
                    if side in inp:
                        inp_dict[side] = inp[side]

        if not inp_dict:
            return f

        def normalize_uv(side, spec):
            if isinstance(spec, dict):
                ux, uy = spec.get("u", 0.0), spec.get("v", 0.0)
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                ux, uy = spec[0], spec[1]
            else:
                ux, uy = (spec, 0.0) if side in ("l", "r") else (0.0, spec)

            def to_fun(val):
                if callable(val):
                    return val
                c = jnp.asarray(val, dtype=jnp.float32)
                return lambda I, J: jnp.broadcast_to(c, I.shape)

            return to_fun(ux), to_fun(uy)

        for side, spec in inp_dict.items():
            if side not in self.params:
                continue

            ux_fun, uy_fun = normalize_uv(side, spec)

            p = self.params[side]
            k_idx = p["bb"]

            if side in ("t", "b"):
                i_idx = p["int"]["i"]
                j = p["int"]["j"]
                K, I = jnp.meshgrid(k_idx, i_idx, indexing="ij")
                J = jnp.full_like(I, j)
            else:
                j_idx = p["int"]["j"]
                i = p["int"]["i"]
                K, J = jnp.meshgrid(k_idx, j_idx, indexing="ij")
                I = jnp.full_like(J, i)

            ux = jnp.asarray(ux_fun(I, J), dtype=jnp.float32)
            uy = jnp.asarray(uy_fun(I, J), dtype=jnp.float32)

            ex = self.ex[k_idx][:, None]
            ey = self.ey[k_idx][:, None]
            eu = ex * ux + ey * uy
            term = 6.0 * self.wt[k_idx][:, None] * rho[I, J] * eu

            new_vals = ftemp[K, I, J] - term

            bb = self.bounce[K]

            free_cell = ~self.mask[I, J]
            f = f.at[bb, I, J].set(jnp.where(free_cell, new_vals, f[bb, I, J]))

        return f

    @partial(jax.jit, static_argnums=0)
    def macroscopic(
        self, f: jnp.ndarray = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rho = jnp.sum(f, axis=0)
        inv_rho = jnp.where(rho > 1e-8, 1.0 / rho, 0.0)
        u = jnp.sum(f * self.ex[:, None, None], axis=0) * inv_rho
        v = jnp.sum(f * self.ey[:, None, None], axis=0) * inv_rho

        return u, v, rho

    @partial(jax.jit, static_argnums=0)
    def pressure(self, rho: jnp.ndarray) -> jnp.ndarray:
        return self.c * rho

    @partial(jax.jit, static_argnums=0)
    def step(
        self,
        f: jnp.ndarray = None,
        u: jnp.ndarray = None,
        v: jnp.ndarray = None,
        rho: jnp.ndarray = None,
        p: jnp.ndarray = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        feq = self.equilibrium(u, v, rho)
        fcol = self.collide(f, feq)
        fstr, ftemp = self.stream(fcol)
        fper = self.periodic(fstr)
        fbb = self.bounce_back(fper, ftemp)
        fneu = self.neumann_bc(fbb, ftemp, rho)
        u, v, rho = self.macroscopic(fneu)
        p = self.pressure(rho)

        return fneu, u, v, rho, p

    def save(
        self,
        dir: str,
        export: str = "npy",
        iteration: int = 0,
        u: jnp.ndarray = None,
        v: jnp.ndarray = None,
        rho: jnp.ndarray = None,
        p: jnp.ndarray = None,
        mask: jnp.ndarray = None,
        conditions: dict = {},
    ) -> None:
        name = os.path.join(dir, f"{self.prefix}{iteration:07d}.{export}")

        if export == "npy":
            jnp.save(
                name,
                {
                    "u": jnp.array(u),
                    "v": jnp.array(v),
                    "rho": jnp.array(rho),
                    "pressure": p,
                    "conditions": conditions,
                    "mask": mask,
                },
            )
        elif export == "dat":
            unp = jnp.asarray(u)
            vnp = jnp.asarray(v)
            rhonp = jnp.asarray(rho)
            pressure = jnp.asarray(p)
            masknp = jnp.asarray(mask, dtype=bool)

            nx, ny = unp.shape

            I, J = jnp.indices((nx, ny))

            out = jnp.column_stack(
                [
                    I.ravel(),
                    J.ravel(),
                    unp.ravel(),
                    vnp.ravel(),
                    rhonp.ravel(),
                    pressure.ravel(),
                    masknp.ravel().astype(int),
                ]
            )

            header_lines = ["# X Y U V RHO PRESSURE MASK"]
            header_lines += [f"# {k}: {v}" for k, v in conditions.items()]
            header = "\n".join(header_lines)

            np.savetxt(
                name,
                out,
                fmt=["%d", "%d", "%.6e", "%.6e", "%.6e", "%.6e", "%d"],
                header=header,
                comments="",
            )
        else:
            raise ValueError(f"Unsupported format ({export}). Use .npy or .dat.")

    def difference(
        self, type: str = "p", indexs: tuple[int, int] = (1, -1), data: jnp.ndarray = None
    ) -> float:
        base = {"p": self.p, "d": self.rho, "v": jnp.sqrt(self.u**2 + self.v**2)}

        if type not in base.keys():
            return "Unknow type"

        d = base[type] if data is None else data

        start = d[indexs[0]][~self.mask[indexs[0]]]
        end = d[indexs[1]][~self.mask[indexs[1]]]

        return jnp.mean(end) - jnp.mean(start)

    def porisity(
        self, dimensions: slice | tuple[slice, ...] = (slice(1, -1), slice(1, -1))
    ):
        return self.mask[dimensions[::-1]]

    def run(
        self,
        steps: int = 5001,
        save: int = 100,
        dir=os.path.join(
            os.path.dirname(os.path.abspath(__import__("__main__").__file__)), "results"
        ),
        export: str = "npy",
        saving: bool = True,
        plotting: bool = True,
        dimensions: slice | tuple[slice, ...] = (slice(1, -1), slice(1, -1)),
    ):
        save_path = os.path.join(dir, export)

        os.makedirs(save_path, exist_ok=True)

        f, mask, u, v, rho, p = self.f, self.mask, self.u, self.v, self.rho, self.p

        for it in tqdm(range(self.continue_iterations, self.continue_iterations + steps)):
            f, u, v, rho, p = self.step(f, u, v, rho, p)

            if saving and it % save == 0:
                self.conditions["iteration"] = it

                self.save(save_path, export, it, u, v, rho, p, mask, self.conditions)

        if plotting:
            print("Plotting...")

            for it in range(
                self.continue_iterations, self.continue_iterations + steps, save
            ):
                plot_path = os.path.join(save_path, f"{it:07d}.{export}")

                plotter(plot_path, rewrite=True, dimensions=dimensions)

        self.f, self.u, self.v, self.rho = f, u, v, rho

    @classmethod
    def load_simulation(
        cls,
        file_dir: str = "0000000.npy",
        conditions: dict = None,
        mask: jnp.ndarray = None,
        prefix: str = None,
        continue_iteration: bool = True,
    ):
        X, Y, u, v, rho, pressure, load_conditions, load_mask = load_data(
            os.path.join(
                os.path.dirname(os.path.abspath(__import__("__main__").__file__)),
                file_dir,
            )
        )

        if conditions is not None:
            load_conditions.update(conditions)

        if mask is not None:
            load_mask = mask

        return cls(
            conditions=load_conditions,
            mask=load_mask,
            u=u.T,
            v=v.T,
            rho=rho.T,
            pressure=pressure.T,
            prefix=prefix,
            continue_iteration=(
                int(load_conditions["iteration"]) if continue_iteration else 0
            ),
        )


def plotter(
    dir: str = "0000000.npy",
    save_dir=None,
    rewrite: bool = False,
    dimensions: slice | tuple[slice, ...] = (slice(1, -1), slice(1, -1)),
    velocity: bool = True,
    density: bool = True,
    vorticity: bool = True,
    pressure: bool = True,
) -> None:
    def create_plot(
        file_path, X, Y, data, u=None, v=None, title: str = "", label: str = ""
    ):
        width = X.max() - X.min()
        height = Y.max() - Y.min()
        aspect_ratio = width // height

        base_height = 6

        figsize = (aspect_ratio * base_height, base_height)

        fig, ax = plt.subplots(figsize=figsize)

        mesh = ax.pcolormesh(X, Y, data, shading="auto", cmap="rainbow")

        fig.colorbar(mesh, ax=ax, label=label)

        if u is not None and v is not None:
            ax.streamplot(X, Y, u, v, density=1, color="k", linewidth=0.7, arrowsize=0.5)

        if mask is not None:
            black_cmap = ListedColormap(["none", "white"])
            ax.pcolormesh(X, Y, mask, shading="auto", cmap=black_cmap, vmin=0, vmax=1)

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        ax.set_xlim(X.min() + 1, X.max() - 1)
        ax.set_ylim(Y.min() + 1, Y.max() - 1)

        # ax.set_aspect('auto')
        ax.set_aspect("equal", "box")

        fig.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    extension = Path(dir).suffix.lstrip(".")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__import__("__main__").__file__)),
        "results",
        extension,
        f"{dir}",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dir/File not exsist: {path}")

    save = (
        os.path.join(
            os.path.dirname(os.path.abspath(__import__("__main__").__file__)),
            "results",
            save_dir,
        )
        if save_dir
        else os.path.join(
            os.path.dirname(os.path.abspath(__import__("__main__").__file__)),
            "results",
            "plots",
        )
    )

    files = (
        [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(f".{extension}")
        ]
        if os.path.isdir(path)
        else [path] if path.endswith(f".{extension}") else []
    )

    if len(files) == 0:
        raise FileNotFoundError(f"Not .{extension} file(s) founded")

    os.makedirs(save, exist_ok=True)

    for file in files:
        base = Path(os.path.basename(file)).stem

        cache_path = os.path.join(save, base)

        os.makedirs(cache_path, exist_ok=True)

        X, Y, u, v, rho, p, conditions, mask = load_data(file)

        base = Path(cache_path).with_suffix("")

        os.makedirs(os.path.join(save, base), exist_ok=True)

        ny, nx = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        X = X[dimensions[::-1]]
        Y = Y[dimensions[::-1]]
        u = u[dimensions[::-1]]
        v = v[dimensions[::-1]]
        rho = rho[dimensions[::-1]]
        p = p[dimensions[::-1]]
        mask = mask[dimensions[::-1]]

        remove = ["nx", "ny", "walls", "periodic", "input"]
        cache = copy.deepcopy(conditions)

        for key in remove:
            cache.pop(key, None)

        remove = ["nx", "ny", "walls", "periodic", "input"]
        cache = conditions.copy()

        for key in remove:
            cache.pop(key, None)

        name = f"{', '.join([*[f'{k}={cache[k]}' for k in cache]])} ({conditions['nx']}, {conditions['ny']})"

        velocity_path = os.path.join(base, "Velocity.png")
        voritcity_path = os.path.join(base, "Vorticity.png")
        density_path = os.path.join(base, "Density.png")
        pressure_path = os.path.join(base, "Pressure.png")

        if velocity and (rewrite or not os.path.exists(velocity_path)):
            data = np.sqrt(u**2 + v**2)
            create_plot(velocity_path, X, Y, data, u, v, name, "Velocity")

        if vorticity and (rewrite or not os.path.exists(voritcity_path)):
            data = np.gradient(v, axis=1) - np.gradient(u, axis=0)
            create_plot(voritcity_path, X, Y, data, None, None, name, "Vorticity")

        if density and (rewrite or not os.path.exists(density_path)):
            create_plot(density_path, X, Y, rho, None, None, name, "Density")

        if pressure and (rewrite or not os.path.exists(pressure_path)):
            create_plot(pressure_path, X, Y, p, None, None, name, "Pressure")
