"""CLI для моделирования парного взаимодействия двух частиц."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from particle_model import simulate


def save_results(outdir: Path, steps: int, dt: float, initial_distance: float, speed: float):
    outdir.mkdir(parents=True, exist_ok=True)
    results = {name: simulate(name, steps, dt, initial_distance, speed) for name in ("euler", "verlet")}

    # Траектории velocity-Verlet
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, label in enumerate(("Частица 1", "Частица 2")):
        p = results["verlet"].positions[:, j, :]
        ax.plot(p[:, 0], p[:, 1], label=label, linewidth=1.6)
        ax.scatter(p[0, 0], p[0, 1], s=35)
    ax.set(title="Траектории частиц (velocity-Verlet)", xlabel="x, усл. ед.", ylabel="y, усл. ед.")
    ax.axis("equal"); ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir/"trajectories.png", dpi=180); plt.close(fig)

    # Траектории метода Эйлера (для сравнения)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j, label in enumerate(("Частица 1", "Частица 2")):
        p = results["euler"].positions[:, j, :]
        ax.plot(p[:, 0], p[:, 1], label=label, linewidth=1.6)
        ax.scatter(p[0, 0], p[0, 1], s=35)
    ax.set(title="Траектории частиц (метод Эйлера)", xlabel="x, усл. ед.", ylabel="y, усл. ед.")
    ax.axis("equal"); ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(outdir/"trajectories_euler.png", dpi=180); plt.close(fig)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, result in results.items():
        rel = np.abs((result.energy-result.energy[0]) / max(abs(result.energy[0]), 1e-12))
        ax.semilogy(result.time, np.maximum(rel, 1e-16), label=name)
    ax.set(title="Относительная ошибка полной энергии", xlabel="Время, усл. ед.", ylabel="|E(t) − E(0)| / |E(0)|")
    ax.grid(True, which="both", alpha=.3); ax.legend(title="Метод"); fig.tight_layout()
    fig.savefig(outdir/"energy_error.png", dpi=180); plt.close(fig)
    with (outdir/"energy_history.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("time,euler_energy,verlet_energy\n")
        for i,t in enumerate(results["verlet"].time): f.write(f"{t:.8g},{results['euler'].energy[i]:.10g},{results['verlet'].energy[i]:.10g}\n")
    with (outdir/"summary.txt").open("w", encoding="utf-8") as f:
        for name,r in results.items():
            drift=np.max(np.abs(r.energy-r.energy[0]))
            f.write(f"{name}: initial_energy={r.energy[0]:.8g}; max_abs_energy_drift={drift:.8g}\n")
    print(f"Готово. Результаты сохранены в: {outdir.resolve()}")
    for name,r in results.items(): print(f"{name}: E0={r.energy[0]:.6g}; max |E-E0|={np.max(np.abs(r.energy-r.energy[0])):.6g}")


def main():
    parser=argparse.ArgumentParser(description="Модель двух частиц с потенциалом Леннарда–Джонса")
    parser.add_argument("--steps",type=int,default=5000,help="число временных шагов")
    parser.add_argument("--dt",type=float,default=0.002,help="шаг времени")
    parser.add_argument("--distance",type=float,default=1.35,help="начальное расстояние между частицами")
    parser.add_argument("--speed",type=float,default=0.15,help="начальная скорость каждой частицы")
    parser.add_argument("--out",type=Path,default=Path("results"),help="каталог результатов")
    a=parser.parse_args(); save_results(a.out,a.steps,a.dt,a.distance,a.speed)
if __name__=="__main__": main()
