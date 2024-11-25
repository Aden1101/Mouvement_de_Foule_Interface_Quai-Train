import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def run_simulation(
    shared_data, num_agents=100, barrier_width=0.15, g=0.1, max_time=10.0
):
    # Paramètres
    L = 1.0  # Taille du domaine
    dt = 0.09  # Intervalle de temps entre les frames
    T = max_time
    sigma = 0.005
    repulsion_strength = 0.001
    collision_distance = 0.05
    blue_distance = 0
    all_blue_crossed = False
    current_time = 0.0  # Timer initialisé à 0

    # Initialisation des agents
    np.random.seed(0)
    agents_pos = np.zeros((num_agents, 2))
    agents_side = np.zeros(num_agents)
    agent_sizes = np.random.uniform(10, 25, num_agents)

    for i in range(num_agents):
        if i < num_agents // 2:
            agents_pos[i, 0] = np.random.uniform(-0.4, -0.1)
            agents_pos[i, 1] = np.random.uniform(-L / 4, L / 4)
            agents_side[i] = 1
        else:
            agents_pos[i, 0] = np.random.uniform(0.2, 0.4)
            agents_pos[i, 1] = np.random.uniform(-L / 4, L / 4)
            agents_side[i] = -1

    # Fonction pour mettre à jour les positions
    def update_agents(agents_pos, agents_side, allow_blue_movement):
        new_positions = []
        for i, (x, y) in enumerate(agents_pos):
            if agents_side[i] == 1 and not allow_blue_movement:
                new_positions.append([x, y])
                continue

            target_x = (
                0 if abs(y) > barrier_width / 2 else (L if agents_side[i] == 1 else -L)
            )

            Vx = g * (target_x - x)
            Vy = -g * (y / abs(y)) if abs(y) > barrier_width / 2 else 0

            repulsion = np.zeros(2)
            for other_pos in agents_pos:
                if np.array_equal([x, y], other_pos):
                    continue
                distance = np.linalg.norm([x, y] - other_pos)
                if distance < collision_distance:
                    repulsion += repulsion_strength * ([x, y] - other_pos) / distance**2

            dx = dt * (Vx + repulsion[0]) + sigma * np.random.randn() * np.sqrt(dt)
            dy = dt * (Vy + repulsion[1]) + sigma * np.random.randn() * np.sqrt(dt)

            new_x = np.clip(x + dx, -L, L)
            new_y = np.clip(y + dy, -L, L)
            new_positions.append([new_x, new_y])

        return np.array(new_positions)

    # Configuration de Matplotlib
    fig, ax = plt.subplots()
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)

    scat = ax.scatter(
        agents_pos[:, 0],
        agents_pos[:, 1],
        c=["blue" if side == 1 else "red" for side in agents_side],
        s=agent_sizes,
    )

    # Ajouter une légende pour le timer
    timer_text = ax.text(
        0.5, 1.05, f"Time: {current_time:.2f}s", transform=ax.transAxes, ha="center"
    )

    def animate(n):
        nonlocal agents_pos, all_blue_crossed, current_time

        # Mettre à jour le timer
        current_time += dt
        timer_text.set_text(f"Time: {current_time:.2f}s")

        # Condition d'arrêt basée sur le temps maximum
        if current_time >= max_time:
            ani.event_source.stop()

        # Condition d'arrêt : Tous les rouges et bleus ont traversé
        all_red_crossed = np.all(agents_pos[agents_side == -1, 0] < -blue_distance)
        agents_pos = update_agents(
            agents_pos, agents_side, allow_blue_movement=all_red_crossed
        )
        scat.set_offsets(agents_pos)

        if not all_blue_crossed:
            all_blue_crossed = np.all(agents_pos[agents_side == 1, 0] > 0)

        if all_red_crossed and all_blue_crossed:
            ani.event_source.stop()

    # Ajout des barrières
    ax.plot([-0.05, -0.05], [-L, -barrier_width / 2], color="black", linewidth=2)
    ax.plot([-0.05, -0.05], [barrier_width / 2, L], color="black", linewidth=2)

    ani = animation.FuncAnimation(fig, animate, frames=int(T / dt), interval=50)
    plt.show()

    # À la fin, enregistrer les données dans `shared_data`
    shared_data["positions"] = agents_pos.tolist()
    shared_data["all_blue_crossed"] = all_blue_crossed
    shared_data["all_red_crossed"] = np.all(
        agents_pos[agents_side == -1, 0] < -blue_distance
    )
    shared_data["final_time"] = current_time
