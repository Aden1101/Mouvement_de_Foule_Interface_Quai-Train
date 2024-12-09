import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.Simulation import Agent, TrainStationSimulation
from matplotlib.animation import FuncAnimation
import csv
import os


# Largeur du trou dans la barrière
def run_simulation(
    simul,
    shared_data,
    steps=100,
    dt=0.02,
):
    """
    Effectue la simulation et retourne les temps nécessaires pour chaque équipe.

    delay_after_crossing: Temps d'attente supplémentaire après que tous les agents ont traversé.
    """
    positions = []
    blue_cross_time = None
    red_cross_time = None
    time = 0

    for step in range(steps):
        simul.update_agents(dt)
        time += dt
        positions.append([agent.position.copy() for agent in simul.agents])

        if blue_cross_time is None and simul.all_blues_crossed:
            blue_cross_time = step * dt

        if red_cross_time is None and simul.are_all_reds_crossed():
            red_cross_time = step * dt

        if simul.all_blues_crossed and simul.are_all_reds_crossed():
            break

    return blue_cross_time, red_cross_time, positions


def save_simulation_to_csv(file_name, results):
    """
    Saves the results of a simulation to a CSV file.
    Appends to the file if it already exists; creates a new file if it does not.

    Args:
        file_name (str): Name of the CSV file.
        results (list of dict): List of simulation results.
    """
    file_exists = os.path.exists(file_name)  # Check if the file already exists
    mode = "a" if file_exists else "w"  # Append if file exists, write otherwise
    header = not file_exists  # Write header if file does not exist

    with open(file_name, mode=mode, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Simulation",
                "Nb_agents",
                "Alpha",
                "Beta",
                "Blue_time",
                "Red_time",
                "Final_time",
            ],
        )
        if header:
            writer.writeheader()
        writer.writerows(results)


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
        color="black",
        label="Barrière",
    )
    ax.plot(
        [simulation.barrier_position, simulation.barrier_position],
        [
            simulation.area_size[1] / 2 + simulation.barrier_width,
            simulation.area_size[1],
        ],
        color="black",
    )

    plt.legend()

    anim = FuncAnimation(
        fig, update, frames=len(positions), interval=interval, blit=True
    )
    plt.title("Simulation Quai/Train avec Barrière")
    plt.show()


def launch_simulation(
    nbr_agent, shared_data, alpha, beta, save_file=None, sim_number=1
):
    if "results" not in shared_data:  # Si on fait plusieurs simulations par exemple
        shared_data["results"] = []
    simul = TrainStationSimulation(
        nbr_agent,
        door_position=[(-5, 5), (15, 5), (9, 8), (9, 2)],
        max_time=20,
        alpha_value=alpha,
        beta_value=beta,
    )
    blue_time, red_time, positions = run_simulation(
        simul, shared_data, steps=500, dt=0.05
    )
    print(
        f"Simulation {sim_number}: Nombre de personnes: {nbr_agent}, Temps de descente: {blue_time:.2f}s, Temps de montée: {red_time:.2f}s"
    )
    # Sauvegarde les résultats dans shared_data pour les statistiques
    # Récupérer la liste existante, la modifier localement, puis la réassigner
    results = shared_data["results"]  # Copie locale
    results.append(
        {
            "Simulation": sim_number,
            "Nb_agents": nbr_agent,
            "Alpha": alpha,
            "Beta": beta,
            "Blue_time": blue_time,
            "Red_time": red_time,
            "Final_time": (
                blue_time + red_time
                if blue_time is not None and red_time is not None
                else None
            ),
        }
    )
    shared_data["results"] = results  # Réassigner à shared_data

    animate_simulation(simul, positions)

    # Sauvegarde dans un fichier CSV si nécessaire
    if not save_file:
        save_file = "simulation_results.csv"

    save_simulation_to_csv(save_file, shared_data["results"])

    return shared_data


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
