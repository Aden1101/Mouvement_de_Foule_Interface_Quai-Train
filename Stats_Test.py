import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Manager
from model.animation import launch_simulation  # Ton code existant
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_cdf(
    csv_path: str,
    param: str,
    time_col: str,
    x_label: str,
    title_prefix: str = "CDF",
    title_suffix: str = "",
):
    df = pd.read_csv(csv_path)
    if param not in df or time_col not in df:
        raise ValueError(f"Colonnes manquantes : {param}, {time_col}")

    values = np.sort(df[param].dropna().unique())
    norm = colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))
    for v in values:
        times = np.sort(df.loc[df[param] == v, time_col].dropna())
        if times.size:
            cdf = np.arange(1, len(times) + 1) / len(times)
            ax.plot(times, cdf, label=f"{param} = {v}", color=cmap(norm(v)))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=f"Valeur de {param}")
    ax.set(xlabel=x_label, ylabel="CDF",
           title=f"{title_prefix} {x_label.lower()} selon {param} {title_suffix}")
    ax.grid(True)
    fig.tight_layout()
    plt.show()

def run_batch(variable_name, values, nbr_agent, n_runs, time_limit,
              output_file, **fixed):
    sim_id = 1
    with Manager() as manager:
        shared = manager.dict()
        for val in values:
            for _ in range(n_runs):
                params = {variable_name.lower(): val, **fixed}
                launch_simulation(
                    nbr_agent=nbr_agent,
                    shared_data=shared,
                    save_file=output_file,
                    sim_number=sim_id,
                    show_animation=False,
                    time_limit=time_limit,
                    extra_info={variable_name: val},
                    **params
                )
                sim_id += 1



if __name__ == "__main__":
    # --- PARAMÈTRES EXPÉRIMENTAUX ---
    # alphas_to_test = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    alphas_to_test = [0.5, 0.7, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 10, 12]
    betas_to_test = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    gammas_to_test = [0.0, 0.001, 0.003, 0.005, 0.007, 0.01, 0.02]

    agents_per_team = 27
    simulations_per_alpha = 20
    time_limit_sec = 60.0
    output_csv = "alpha_cdf_results.csv"

    # --- LANCEMENT DES SIMULATIONS ---
    """
    run_batch("Alpha", alphas_to_test, agents_per_team, simulations_per_alpha,
          time_limit_sec, "alpha_cdf_results.csv", beta=2.0, gamma_zigzag=0.005)

    )"""

    plot_cdf("alpha_cdf_results.csv", param="Alpha", time_col="Final_time", x_label="Temps final (s)")
    plot_cdf("alpha_cdf_results.csv", param="Alpha", time_col="Red_95_time", x_label="Temps pour 95 % des rouges (s)")


    # Pour tester Beta :
    # run_batch_simulations_varying_beta(betas_to_test, agents_per_team, simulations_per_alpha, output_file="beta_results.csv")

    # Pour afficher les graphes :
    # plot_cdf_by_param("beta_results.csv", param_name="Beta")
    # plot_cdf_95_time_by_param("beta_results.csv", param_name="Beta")

    # Pour tester Gamma :
    # run_batch_simulations_varying_gamma(gammas_to_test, agents_per_team, simulations_per_alpha, output_file="gamma_results.csv")

    # Pour afficher les graphes :
    # plot_cdf_by_param("gamma_results.csv", param_name="Gamma")
    # plot_cdf_95_time_by_param("gamma_results.csv", param_name="Gamma")

