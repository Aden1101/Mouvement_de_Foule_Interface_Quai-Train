import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Manager
from model.animation import launch_simulation  # Ton code existant
import matplotlib.cm as cm
import matplotlib.colors as colors


def plot_cdf_final_time_by_alpha(csv_path):
    df = pd.read_csv(csv_path)
    if "Alpha" not in df.columns or "Final_time" not in df.columns:
        print("Erreur : colonnes manquantes.")
        return

    alphas = sorted(df["Alpha"].unique())
    norm = colors.Normalize(vmin=min(alphas), vmax=max(alphas))
    cmap = cm.viridis  # Choisis un colormap, tu peux essayer plasma, inferno, coolwarm, etc.

    fig, ax = plt.subplots(figsize=(10, 6)) 

    for alpha in alphas:
        times = df[df["Alpha"] == alpha]["Final_time"].values
        times = np.sort(times)
        cdf = np.arange(1, len(times) + 1) / len(times)
        plt.plot(times, cdf, label=f"α = {alpha}", color=cmap(norm(alpha)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Nécessaire pour la légende
    cbar = plt.colorbar(sm, ax=ax) 
    cbar.set_label("Valeur de α")

    plt.xlabel("Temps final (s)")
    plt.ylabel("Fonction de répartition cumulée (CDF)")
    plt.title("CDF du temps final pour différentes valeurs de α")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cdf_red_95_time(csv_path):
    df = pd.read_csv(csv_path)
    if "Alpha" not in df.columns or "Red_95_time" not in df.columns:
        print("Erreur : colonnes manquantes (Red_95_time).")
        return

    alphas = sorted(df["Alpha"].unique())
    norm = colors.Normalize(vmin=min(alphas), vmax=max(alphas))
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6)) 

    for alpha in alphas:
        times_95 = df[df["Alpha"] == alpha]["Red_95_time"].dropna().values
        if len(times_95) == 0:
            continue
        times_95 = np.sort(times_95)
        cdf = np.arange(1, len(times_95) + 1) / len(times_95)
        plt.plot(times_95, cdf, label=f"α = {alpha}", color=cmap(norm(alpha)))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax) 
    cbar.set_label("Valeur de α")

    plt.xlabel("Temps où 95% des rouges ont traversé (s)")
    plt.ylabel("Fonction de répartition cumulée (CDF)")
    plt.title("CDF des temps à 95% des rouges traversés selon α")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_batch_simulations(
    alphas,
    nb_agents,
    num_simulations_per_alpha,
    beta=2.0,
    gamma=0.005,
    time_limit=60.0,
    output_file="alpha_cdf_results.csv",
):
    with Manager() as manager:
        shared_data = manager.dict()
        sim_id = 1
        for alpha in alphas:
            for i in range(num_simulations_per_alpha):
                launch_simulation(
                    nbr_agent=nb_agents,
                    shared_data=shared_data,
                    alpha=alpha,
                    beta=beta,
                    gamma_zigzag=gamma,
                    save_file=output_file,
                    sim_number=sim_id,
                    show_animation=False,
                    time_limit=time_limit,
                )
                sim_id += 1


if __name__ == "__main__":
    # --- PARAMÈTRES EXPÉRIMENTAUX ---
    # alphas_to_test = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    alphas_to_test = [0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10, 12]
    agents_per_team = 27
    simulations_per_alpha = 20
    time_limit_sec = 60.0
    output_csv = "alpha_cdf_results.csv"

    # --- LANCEMENT DES SIMULATIONS ---
    """
    run_batch_simulations(
        alphas=alphas_to_test,
        nb_agents=agents_per_team,
        num_simulations_per_alpha=simulations_per_alpha,
        time_limit=time_limit_sec,
        output_file=output_csv,
    )"""

    # --- AFFICHAGE DU GRAPHIQUE CDF ---
    plot_cdf_final_time_by_alpha(output_csv)

    plot_cdf_red_95_time(output_csv)
