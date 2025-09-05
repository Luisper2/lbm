import os
import time

import jax.numpy as jnp

from lbmpy.lbm import LBM


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
    f = 0.1
    cylinders = 10

    radius = jnp.sqrt((f * l**2) / (2 * jnp.pi))
    radius = 0.125 * l

    f = (2 * jnp.pi * radius**2) / l**2

    buffer = 2 * l

    max_col = (cylinders - 1) // 2 + (
        0.5 if (cylinders % 2 == 0 and cylinders > 0) else 0.0
    )

    nx = int(2 * buffer + 2 * radius + l * max_col)
    ny = l

    I, J = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    mask = jnp.zeros((nx, ny), dtype=bool)

    top = True
    for i in range(cylinders):
        col = i // 2 + (0.5 if i % 2 == 1 else 0.0)

        x = (buffer + radius + col * l) - 1
        y = (3 * l / 4 if top else l / 4) - 1

        mask = mask | ((I - x) ** 2 + (J - y) ** 2 <= radius**2)

        top = not top

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
        "periodic": ["v", "h"],
    }

    simulation = LBM(conditions=conditions, mask=mask)
    simulation.run(
        steps=2501,
        save=100,
        export=export,
        plotting=True,
    )

    col = cylinders // 2 + (0.5 if i % 2 == 1 else 0.0)
    x = (buffer + radius + col * l) - 1

    print(simulation.difference(type="v"))
    q_in, q_out = flux_x(simulation.u, simulation.mask)
    print("Q_in, Q_out:", q_in, q_out)
    print(q_out - q_in)
    print(jnp.abs(q_out - q_in) / q_in * 100, "%")

    print(f"Done ({(time.perf_counter() - start):.3f}s)")
