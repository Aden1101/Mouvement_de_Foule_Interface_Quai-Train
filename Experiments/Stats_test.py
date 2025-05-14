from multiprocessing import Manager
from model.animation import launch_simulation

alphas = [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
N = 1  # répétitions par point
beta = 2.0
gamma = 0.005
manager = Manager()
shared = manager.dict()

for a in alphas:
    for k in range(N):
        launch_simulation(
            nbr_agent=20,
            shared_data=shared,
            alpha=a,
            beta=beta,
            gamma_zigzag=gamma,
            save_file="sweep_alpha.csv",
            sim_number=len(shared.get("results", [])) + 1,
            show_animation=False,
            time_limit=60,
        )
