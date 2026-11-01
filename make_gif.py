import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from particle_model import simulate

def create_animation(method="verlet", filename="particle_dynamics.gif"):
    # Запускаем симуляцию на 1000 шагов для красоты
    result = simulate(method=method, steps=1000, dt=0.002)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Динамика частиц ({method})", fontsize=14)
    ax.set_xlabel("x, усл. ед.")
    ax.set_ylabel("y, усл. ед.")
    
    # Линия, соединяющая частицы, и точки
    line, = ax.plot([], [], 'o-', lw=1.5, markersize=8)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        # Берем каждый 5-й кадр, чтобы GIF не был слишком тяжелым
        step = i * 5
        if step >= len(result.positions): 
            step = len(result.positions) - 1
        pos = result.positions[step]
        # pos[0] - координаты первой, pos[1] - второй
        line.set_data([pos[0, 0], pos[1, 0]], [pos[0, 1], pos[1, 1]])
        return line,
    
    frames = len(result.positions) // 5
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=30, blit=True)
    
    # Сохраняем в GIF
    anim.save(filename, writer=PillowWriter(fps=30))
    plt.close(fig)
    print(f"Анимация сохранена в {filename}")

def plot_evolution_snapshots(filename="evolution_snapshots.png"):
    """Сохраняет 4 кадра эволюции системы в одном рисунке."""
    result = simulate(method="verlet", steps=1000, dt=0.002)
    
    steps = [0, 250, 500, 999]
    titles = ["(а) t = 0", "(б) t = 0.5", "(в) t = 1.0", "(г) t = 2.0"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for ax, step, title in zip(axes, steps, titles):
        pos = result.positions[step]
        ax.plot(pos[:, 0], pos[:, 1], 'o', markersize=10, color='C0')
        ax.plot(pos[:, 0], pos[:, 1], 'k--', alpha=0.5)
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        if ax != axes[0]:
            ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()
    print(f"Снапшоты сохранены в {filename}")

if __name__ == "__main__":
    create_animation("verlet", "particle_dynamics.gif")
    plot_evolution_snapshots("evolution_snapshots.png")