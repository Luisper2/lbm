import os
import time

import jax.numpy as jnp

from lbmpy.lbm import LBM


def build_periodic_cylinder_mask(nx, ny, l, h, n_cylinders, radius):
    """
    nx, ny: interior domain size (same as LBM.conditions['nx'], ['ny'])
    l, h: lattice spacings used to place cylinder rows/columns (in grid cells)
    n_cylinders: cylinders per row (staggered 2-row layout)
    radius: cylinder radius in grid cells

    Returns: mask with shape (nx+2, ny+2), including periodic halos.
    """

    # Interior index grid: i in [1..nx], j in [1..ny]
    I, J = jnp.meshgrid(jnp.arange(1, nx + 1), jnp.arange(1, ny + 1), indexing="ij")

    def pbc_delta(a, b, L):
        # Minimum-image displacement on periodic domain of length L
        return jnp.mod(a - b + 0.5 * L, L) - 0.5 * L

    # Build interior mask
    mask_int = jnp.zeros((nx, ny), dtype=bool)

    top = True
    for k in range(n_cylinders):
        # Centers at 0.5*l, 1.5*l, ... along x; staggered rows at 0.5*h and 1.5*h
        cx = (k + 0.5) * l
        cy = 1.5 * h if top else 0.5 * h

        dx = pbc_delta(I, cx, nx)
        dy = pbc_delta(J, cy, ny)
        mask_int = mask_int | (dx**2 + dy**2 <= radius**2)

        top = not top

    # Allocate full mask with halos (+2 in each dim) and inject interior
    mask = jnp.zeros((nx + 2, ny + 2), dtype=bool)
    mask = mask.at[1 : nx + 1, 1 : ny + 1].set(mask_int)

    # Periodic halos: copy opposite interior edges
    # Left/right halos
    mask = mask.at[0, 1 : ny + 1].set(mask_int[-1, :])  # i=0  <- i=nx
    mask = mask.at[nx + 1, 1 : ny + 1].set(mask_int[0, :])  # i=nx+1 <- i=1
    # Bottom/top halos
    mask = mask.at[1 : nx + 1, 0].set(mask_int[:, -1])  # j=0  <- j=ny
    mask = mask.at[1 : nx + 1, ny + 1].set(mask_int[:, 0])  # j=ny+1 <- j=1
    # Corners
    mask = mask.at[0, 0].set(mask_int[-1, -1])
    mask = mask.at[0, ny + 1].set(mask_int[-1, 0])
    mask = mask.at[nx + 1, 0].set(mask_int[0, -1])
    mask = mask.at[nx + 1, ny + 1].set(mask_int[0, 0])

    return mask


def flux_x(u, mask):
    ux = u[1:-1, 1:-1]
    m = ~mask[1:-1, 1:-1]
    q_in = jnp.sum(ux[0, :] * m[0, :])
    q_out = jnp.sum(ux[-1, :] * m[-1, :])
    return float(q_in), float(q_out)


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    start = time.perf_counter()

    l = 56
    radius = l / 4  # / jnp.sqrt(2)  # or l / jnp.sqrt(2)

    n_cylinders = 4
    h = l

    # For a periodic grid, do not include the duplicated end point
    nx = n_cylinders * l
    ny = 2 * h

    mask = build_periodic_cylinder_mask(nx, ny, l, h, n_cylinders, radius)

    re = 1
    velocity = 0.01
    nu = velocity * (2 * radius) / re
    tau = 3 * nu + 0.5

    export = "dat"

    conditions = {
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "walls": [],
        "periodic": ["h", "v"],
        "body_force": (1e-4, 0.0),
    }

    simulation = LBM(conditions=conditions, mask=mask)
    simulation.run(
        steps=1001,
        save=500,
        export=export,
        plotting=True,
    )

    print(simulation.difference(type="v"))
    q_in, q_out = flux_x(simulation.u, simulation.mask)
    print("Q_in, Q_out:", q_in, q_out)
    print(q_out - q_in)
    print(jnp.abs(q_out - q_in) / q_in * 100, "%")

    print(f"Done ({(time.perf_counter() - start):.3f}s)")
