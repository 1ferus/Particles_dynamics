"""Модель парного взаимодействия частиц с потенциалом Леннарда — Джонса.

Система интегрируется в двумерном пространстве двумя схемами:
explicit Euler и velocity-Verlet. Единицы безразмерные: m = epsilon = sigma = 1.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResult:
    time: np.ndarray
    positions: np.ndarray  # (steps, 2 particles, 2 coordinates)
    velocities: np.ndarray
    energy: np.ndarray
    method: str


def lj_force(positions: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Возвращает силы на две частицы; учитывает только парное взаимодействие."""
    p = np.asarray(positions, dtype=float)
    if p.shape != (2, 2):
        raise ValueError("positions должны иметь форму (2, 2)")
    if epsilon <= 0 or sigma <= 0:
        raise ValueError("epsilon и sigma должны быть положительными")
    delta = p[0] - p[1]
    r2 = float(np.dot(delta, delta))
    if r2 < 1e-12:
        raise ValueError("Частицы не могут находиться в одной точке")
    sr2 = sigma * sigma / r2
    sr6 = sr2 ** 3
    sr12 = sr6 ** 2
    f0 = 24.0 * epsilon * (2.0 * sr12 - sr6) / r2
    f = f0 * delta
    return np.array([f, -f])


def potential_energy(positions: np.ndarray, epsilon: float = 1.0, sigma: float = 1.0) -> float:
    p = np.asarray(positions, dtype=float)
    if p.shape != (2, 2):
        raise ValueError("positions должны иметь форму (2, 2)")
    r = float(np.linalg.norm(p[0] - p[1]))
    if r < 1e-12:
        raise ValueError("Расстояние между частицами должно быть положительным")
    sr6 = (sigma / r) ** 6
    return float(4.0 * epsilon * (sr6 * sr6 - sr6))


def total_energy(positions: np.ndarray, velocities: np.ndarray, mass: float = 1.0,
                 epsilon: float = 1.0, sigma: float = 1.0) -> float:
    v = np.asarray(velocities, dtype=float)
    if mass <= 0:
        raise ValueError("mass должна быть положительной")
    kinetic = 0.5 * mass * float(np.sum(v * v))
    return kinetic + potential_energy(positions, epsilon, sigma)


def simulate(method: str = "verlet", steps: int = 5000, dt: float = 0.002,
             initial_distance: float = 1.35, speed: float = 0.15,
             mass: float = 1.0, epsilon: float = 1.0, sigma: float = 1.0) -> SimulationResult:
    """Запускает двумерную симуляцию двух частиц."""
    if method not in {"euler", "verlet"}:
        raise ValueError("method должен быть 'euler' или 'verlet'")
    if steps < 2 or dt <= 0 or initial_distance <= 0 or mass <= 0 or speed < 0:
        raise ValueError("steps >= 2, dt, distance, mass > 0 и speed >= 0")
    pos = np.array([[-initial_distance / 2, 0.0], [initial_distance / 2, 0.0]], dtype=float)
    vel = np.array([[0.0, speed], [0.0, -speed]], dtype=float)
    positions = np.empty((steps, 2, 2)); velocities = np.empty_like(positions)
    energy = np.empty(steps); time = np.arange(steps) * dt
    force = lj_force(pos, epsilon, sigma)
    for i in range(steps):
        positions[i] = pos; velocities[i] = vel
        energy[i] = total_energy(pos, vel, mass, epsilon, sigma)
        if i == steps - 1: break
        if method == "euler":
            # Явный метод Эйлера: r(n+1)=r(n)+v(n)dt; v(n+1)=v(n)+a(n)dt.
            pos = pos + vel * dt
            vel = vel + force / mass * dt
            force = lj_force(pos, epsilon, sigma)
        else:
            # Velocity-Verlet: второй порядок точности и лучшая энергетическая устойчивость.
            pos_new = pos + vel * dt + 0.5 * force / mass * dt * dt
            force_new = lj_force(pos_new, epsilon, sigma)
            vel = vel + 0.5 * (force + force_new) / mass * dt
            pos, force = pos_new, force_new
    return SimulationResult(time, positions, velocities, energy, method)
