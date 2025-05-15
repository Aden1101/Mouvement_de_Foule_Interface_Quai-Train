import matplotlib.pyplot as plt
from model.Simulation import TrainStationSimulation
from matplotlib.animation import FuncAnimation
import csv
import os
import numpy as np

coeff_time = 3


def run_simulation(
    simul,
    shared_data,
    steps=500,
    dt=0.04,
    # on ajoute deux nouveaux paramètres
    time_limit=60.0,
):
    """
    Effectue la simulation et retourne plusieurs informations, notamment:
    - temps à 95% et 100% de descente (blue)
    - temps à 95% et 100% de montée (red)
    - nb d'agents n'ayant pas traversé à l'instant de fermeture des portes
    - la liste complète des positions, densités, états side, etc.

    steps          : nombre maximal d'itérations
    dt             : pas de temps
    time_limit     : moment où les portes se ferment
    """

    positions = []
    density_history = []
    side_history = []

    # On conserve ici les "moments" auxquels on constate que 95 % / 100 % des agents ont fini
    blue_95_time = None
    blue_cross_time = None  # 100 %
    red_95_time = None
    red_cross_time = None  # 100 %

    # Pour enregistrer le nombre d'agents n’ayant pas encore traversé quand la porte se ferme
    not_crossed_at_close = 0
    door_closed_recorded = (
        False  # Pour éviter de le recalculer à chaque frame une fois qu'on l'a
    )

    time = 0.0

    # Repérage des blues et reds initiaux
    all_agents = simul.agents
    blues = [a for a in all_agents if a.side == 1]  # "descente"
    reds = [a for a in all_agents if a.side == -1]  # "montée"

    total_blue = len(blues)
    total_red = len(reds)

    for step in range(steps):
        # --- Mise à jour du temps et des positions
        simul.update_agents(dt)
        time += dt

        # Stocker l’état pour l’animation
        positions.append([agent.position.copy() for agent in simul.agents])
        density_history.append(simul.density_grid.copy())
        side_history.append([agent.side for agent in simul.agents])

        # --- Calcul du pourcentage d’agents qui ont franchi la barrière
        # Pour "franchir" la barrière, dans le code, on utilise agent.has_crossed = True
        # pour les bleus => side = 1, on les considère "crossed" quand barrier_position - 0.3 est dépassée
        # pour les rouges => side = -1, c’est quand ils sont passés à droite de la barrière, etc.
        # Le code de Simulation gère déjà agent.has_crossed, on s’en sert directement.

        # Nombre de bleus qui ont franchi:
        blues_crossed = sum(1 for a in blues if a.has_crossed)
        reds_crossed = sum(1 for a in reds if a.has_crossed)

        # fraction franchie
        if total_blue > 0:
            frac_blue = blues_crossed / total_blue
        else:
            frac_blue = 1.0  # s'il n'y a aucun agent bleu

        if total_red > 0:
            frac_red = reds_crossed / total_red
        else:
            frac_red = 1.0

        # 95% de descente
        if frac_blue >= 0.95 and blue_95_time is None:
            blue_95_time = coeff_time * time

        # 100% de descente
        if frac_blue >= 1.0 and blue_cross_time is None:
            blue_cross_time = coeff_time * time

        # 95% de montée
        if frac_red >= 0.8 and red_95_time is None:
            red_95_time = coeff_time * time

        # 100% de montée
        if frac_red >= 1.0 and red_cross_time is None:
            red_cross_time = coeff_time * time

        # --- Vérifier instant de fermeture de portes
        if coeff_time * time >= time_limit:
            # On compte combien n'ont pas franchi
            print("atteint")
            not_crossed_at_close = sum(
                1 for ag in all_agents if ag.has_crossed is False
            )
            break

        # --- Vérifier conditions d’arrêt
        # 1) Si tous bleus et tous rouges ont traversé => on arrête
        if simul.all_blues_crossed and simul.are_all_reds_crossed():
            break

    # À la fin, si on n’a jamais calculé un temps, on peut mettre None ou la dernière valeur (time)
    if blue_95_time is None:
        blue_95_time = time_limit
    if blue_cross_time is None:
        blue_cross_time = time_limit
    if red_95_time is None:
        red_95_time = time_limit
    if red_cross_time is None:
        red_cross_time = time_limit

    return (
        blue_95_time,
        blue_cross_time,
        red_95_time,
        red_cross_time,
        not_crossed_at_close,
        positions,
        density_history,
        side_history,
    )


def save_simulation_to_csv(file_name, results):
    """
    Sauvegarde les données de la simulation dans un fichier CSV.
    On ajoute maintenant de nouvelles colonnes pour :
    - Blue_95_time
    - Red_95_time
    - time_limit
    - NotCrossed_atClose
    """
    file_exists = os.path.exists(file_name)
    mode = "a" if file_exists else "w"
    header = not file_exists

    with open(file_name, mode=mode, newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Simulation",
                "Nb_agents",
                "Gamma",
                "Alpha",
                "Beta",
                "Blue_95_time",
                "Blue_time",
                "Red_95_time",
                "Red_time",
                "time_limit",
                "NotCrossed_atClose",
                "Final_time",
            ],
        )
        if header:
            writer.writeheader()
        writer.writerows(results)


def animate_simulation(
    simulation, positions, density_history, side_history, interval=30, dt=0.04
):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Imshow initial
    density_img = ax.imshow(
        density_history[0],
        extent=[0, simulation.area_size[0], 0, simulation.area_size[1]],
        origin="lower",
        cmap="Reds",
        alpha=0.4,
        vmin=0,
        vmax=6,
        interpolation="nearest",
    )

    ax.set_xlim(0, simulation.area_size[0])
    ax.set_ylim(0, simulation.area_size[1])
    ax.set_aspect("equal", adjustable="box")

    # Dessin initial des agents
    circles = [agent.draw(ax) for agent in simulation.agents]

    # Barrière
    ax.plot(
        [simulation.barrier_position, simulation.barrier_position],
        [0, simulation.area_size[1] / 2 - simulation.barrier_width],
        color="black",
        label="Porte du Train",
    )
    ax.plot(
        [simulation.barrier_position, simulation.barrier_position],
        [
            simulation.area_size[1] / 2 + simulation.barrier_width,
            simulation.area_size[1],
        ],
        color="black",
    )

    time_text = ax.text(
        0.05, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=12
    )

    plt.legend()

    def update(frame):
        # 2) Actualiser la densité affichée
        density_img.set_data(density_history[frame])

        # 3) Actualiser la position/couleur des cercles
        for i, circle in enumerate(circles):
            circle.center = positions[frame][i]
            curr_side = side_history[frame][i]
            if curr_side == 1:
                circle.set_color("blue")
            elif curr_side == -1:
                circle.set_color("red")
            elif curr_side == 2:
                circle.set_color("green")
            else:
                circle.set_color("gray")

        # 4) Calculer le temps courant = frame * dt
        current_time = coeff_time * frame * dt
        time_text.set_text(f"Time = {current_time:.2f} s")

        # 5) Retourner tous les objets animés
        return [density_img] + circles + [time_text]

    anim = FuncAnimation(
        fig, update, frames=len(positions), interval=interval, blit=True
    )
    plt.title("Simulation Quai/Train")
    plt.show()


def launch_simulation(
    nbr_agent,
    shared_data,
    alpha,
    beta,
    gamma_zigzag=0.01,
    save_file=None,
    sim_number=1,
    show_animation=True,
    time_limit=60.0,
):
    if "results" not in shared_data:
        shared_data["results"] = []

    simul = TrainStationSimulation(
        nbr_agent,
        max_time=20,
        alpha_value=alpha,
        beta_value=beta,
        gamma_zigzag=gamma_zigzag,
    )

    # --- Lancement de la simulation avec nos nouvelles informations
    (
        blue_95_time,
        blue_time,
        red_95_time,
        red_time,
        not_crossed_at_close,
        positions,
        density_history,
        side_history,
    ) = run_simulation(
        simul,
        shared_data,
        steps=500,
        dt=0.04,
        time_limit=time_limit,
    )

    # On peut définir un "final_time" comme l’instant où tout est franchi
    # (ou la fin du dernier step). Ici, c'est le max entre blue_time et red_time
    final_time = max(blue_time, red_time)

    print(
        f"Simulation {sim_number}: Nb persons: {nbr_agent}, "
        f"Blue_95:{blue_95_time:.2f}s, Blue_100:{blue_time:.2f}s, "
        f"Red_95:{red_95_time:.2f}s, Red_100:{red_time:.2f}s, "
        f"NotCrossedAtClose={not_crossed_at_close}"
    )

    # On enregistre dans results
    results = shared_data["results"]
    results.append(
        {
            "Simulation": sim_number,
            "Nb_agents": nbr_agent,
            "Alpha": alpha,
            "Beta": beta,
            "Gamma": gamma_zigzag,
            "Blue_95_time": blue_95_time,
            "Blue_time": blue_time,
            "Red_95_time": red_95_time,
            "Red_time": red_time,
            "time_limit": time_limit,
            "NotCrossed_atClose": not_crossed_at_close,
            "Final_time": final_time,
        }
    )
    shared_data["results"] = results

    # Affichage animation si demandé
    if show_animation:
        animate_simulation(simul, positions, density_history, side_history)

    # Sauvegarde CSV
    if not save_file:
        save_file = "simulation_results.csv"

    save_simulation_to_csv(save_file, shared_data["results"])
    print(f"Résultats sauvegardés dans {save_file}")

    return shared_data
