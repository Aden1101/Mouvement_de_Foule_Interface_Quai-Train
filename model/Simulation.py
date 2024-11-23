import numpy as np


class SimulationManager:
    def __init__(self, num_agents, barrier_width, collision_distance):
        self.num_agents = num_agents
        self.barrier_width = barrier_width
        self.collision_distance = collision_distance
        self.agents_pos, self.agents_side = self.init_agents()
        self.all_blue_crossed = False

    def init_agents(self):
        agents_pos = np.zeros((self.num_agents, 2))
        agents_side = np.zeros(self.num_agents)
        # Initialize agents here...
        return agents_pos, agents_side

    def update(self):
        # Update agent positions
        pass
