#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os


def load_npy_frame(file_path):
    """
    Carga un frame .npy con [u, v, rho] y devuelve X, Y, u, v, speed, rho.
    """
    u, v, rho = np.load(file_path)
    u = u.T
    v = v.T
    rho = rho.T
    speed = np.sqrt(u**2 + v**2)
    ny, nx = u.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    return X, Y, u, v, speed, rho

def compute_vorticity(u, v):
    """
    Vorticidad 2D: ω = ∂v/∂x - ∂u/∂y
    """
    dvdx = np.gradient(v, axis=1)
    dudy = np.gradient(u, axis=0)
    return dvdx - dudy

def plot_velocity(X, Y, u, v, speed, out_path):
    fig, ax = plt.subplots(figsize=(6,6))
    mesh = ax.pcolormesh(X, Y, speed, shading='auto')
    fig.colorbar(mesh, ax=ax, label='Magnitud de velocidad')
    ax.streamplot(X, Y, u, v, density=1, color='k', linewidth=0.7, arrowsize=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_vorticity(X, Y, u, v, out_path):
    omega = compute_vorticity(u, v)
    fig, ax = plt.subplots(figsize=(6,6))
    mesh = ax.pcolormesh(X, Y, omega, shading='auto')
    fig.colorbar(mesh, ax=ax, label='Vorticidad')
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_density(X, Y, rho, out_path):
    fig, ax = plt.subplots(figsize=(6,6))
    mesh = ax.pcolormesh(X, Y, rho, shading='auto')
    fig.colorbar(mesh, ax=ax, label='Densidad')
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main(DATA_DIR):
    if not os.path.isdir(DATA_DIR):
        print(f"⚠️ No existe la carpeta de datos '{DATA_DIR}'")
        return

    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    if not npy_files:
        print(f"⚠️ No hay archivos .npy en '{DATA_DIR}'")
        return

    for fname in npy_files:
        base = os.path.splitext(fname)[0]
        frame_path = os.path.join(DATA_DIR, fname)
        out_dir = os.path.join('./.cache', base)
        os.makedirs(out_dir, exist_ok=True)

        X, Y, u, v, speed, rho = load_npy_frame(frame_path)

        vel_path = os.path.join(out_dir, f"{base}_velocidad.png")
        plot_velocity(X, Y, u, v, speed, vel_path)
        print(f"✅ Guardada velocidad en {vel_path}")

        vort_path = os.path.join(out_dir, f"{base}_vorticidad.png")
        plot_vorticity(X, Y, u, v, vort_path)
        print(f"✅ Guardada vorticidad en {vort_path}")

        den_path = os.path.join(out_dir, f"{base}_densidad.png")
        plot_density(X, Y, rho, den_path)
        print(f"✅ Guardada densidad en {den_path}")

