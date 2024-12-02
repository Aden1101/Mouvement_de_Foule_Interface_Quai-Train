import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.Simulation import Agent, TrainStationSimulation
from matplotlib.animation import FuncAnimation


# Largeur du trou dans la barrière
def run_simulation(
    simul,
    shared_data,
    steps=100,
    dt=0.02,
):
    """Effectue la simulation et retourne les temps nécessaires pour chaque équipe."""
    positions = []
    blue_cross_time = None
    red_cross_time = None

    for step in range(steps):
        simul.update_agents(dt)
        positions.append([agent.position.copy() for agent in simul.agents])

        if blue_cross_time is None and simul.all_blues_crossed:
            blue_cross_time = step * dt

        if red_cross_time is None and simul.are_all_reds_crossed():
            red_cross_time = step * dt

        if blue_cross_time is not None and red_cross_time is not None:
            break

    return blue_cross_time, red_cross_time, positions


# Visualisation
def animate_simulation(simulation, positions, interval=100):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(2, simulation.area_size[0] - 2)
    ax.set_ylim(2, simulation.area_size[1] - 2)

    # Dessiner les agents initialement
    circles = [agent.draw(ax) for agent in simulation.agents]

    def update(frame):
        for i, circle in enumerate(circles):
            circle.center = positions[frame][i]
        return circles

    # Ajouter la barrière avec un trou
    ax.plot(
        [simulation.barrier_position, simulation.barrier_position],
        [0, simulation.area_size[1] / 2 - simulation.barrier_width],
        label="Barrière",
    )
    ax.plot(
        [simulation.barrier_position, simulation.barrier_position],
        [
            simulation.area_size[1] / 2 + simulation.barrier_width,
            simulation.area_size[1],
        ],
        label="Barrière",
    )

    plt.legend()

    anim = FuncAnimation(
        fig, update, frames=len(positions), interval=interval, blit=True
    )
    plt.title("Simulation Quai/Train avec Barrière")
    plt.show()


def launch_simulation(nbr_agent, shared_data, alpha, beta):
    simul = TrainStationSimulation(
        nbr_agent,
        door_position=[(-5, 5), (15, 5), (8, 8), (8, 2), (6, -2), (6, 2)],
        max_time=20,
        alpha_value=alpha,
        beta_value=beta,
    )
    blue_time, red_time, positions = run_simulation(
        simul, shared_data, steps=500, dt=0.05
    )
    print(
        f"Nombre de personnes: {nbr_agent}, Temps de descente: {blue_time:.2f}s, Temps de montée: {red_time:.2f}s"
    )
    animate_simulation(simul, positions)
    shared_data["final_time"] = blue_time + red_time


# Lancer 10 simulations avec différentes tailles et collecter les temps
"""
results = []
team_size = [20, 30]

for size in team_size:
    simulation = TrainStationSimulation(
        num_agents_per_team=size,
        door_position=[(-5, 5), (15, 5)],
        max_time=20,
    )
    blue_time, red_time, positions = simulation.run_simulation(steps=500, dt=0.05)
    results.append((size, blue_time, red_time))

    # Afficher l'animation pour chaque simulation
    print(
        f"Nombre de personnes: {size}, Temps de descente: {blue_time:.2f}s, Temps de montée: {red_time:.2f}s"
    )
    animate_simulation(simulation, positions)

    shared_data["final_time"] = blue_time + red_time
"""
