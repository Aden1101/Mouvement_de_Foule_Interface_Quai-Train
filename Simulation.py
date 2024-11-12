import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
num_agents = 100  # Total number of agents
T = 10.0  # Total time for animation
L = 1.0  # Domain size
dt = 0.09  # Time step
sigma = 0.005  # Reduced noise intensity (less erratic movement)
barrier_width = 0.15  # Width of the opening in the barrier
g = 0.1  # Attraction strength towards target
repulsion_strength = 0.001  # Strength of repulsion between agents
collision_distance = 0.05  # Minimum allowable distance between agents
blue_distance = 0
all_blue_crossed = False

# Initialize agent positions: half on the left, half on the right
np.random.seed(0)
agents_pos = np.zeros((num_agents, 2))
agents_side = np.zeros(num_agents)
agent_sizes = np.random.uniform(10, 25, num_agents)  # Random sizes for each agent

# Divide agents into two groups
for i in range(num_agents):
    if i < num_agents // 2:
        # Left side agents (Blue), move to the right side
        agents_pos[i, 0] = np.random.uniform(-0.4, -0.1)  # x position
        agents_pos[i, 1] = np.random.uniform(-L / 4, L / 4)  # y position
        agents_side[i] = 1  # Target to the right
    else:
        # Right side agents (Red), move to the left side
        agents_pos[i, 0] = np.random.uniform(0.2, 0.1)  # x position
        agents_pos[i, 1] = np.random.uniform(-L / 4, L / 4)  # y position
        agents_side[i] = -1  # Target to the left


# Define barrier function with a small opening
def barrier_potential(x, y):
    if -barrier_width / 2 < y < barrier_width / 2:
        return 0  # Opening in the barrier
    return 100  # High potential representing the barrier


# Calculate repulsive force from nearby agents
def repulsive_force(pos, all_positions):
    force = np.zeros(2)
    for other_pos in all_positions:
        if np.array_equal(pos, other_pos):
            continue
        distance = np.linalg.norm(pos - other_pos)
        if distance < collision_distance:
            # Apply a repulsive force proportional to the inverse of the distance
            force += repulsion_strength * (pos - other_pos) / distance**2
    return force


# Update function for agents' movement
def update_agents(agents_pos, agents_side, allow_blue_movement):
    new_positions = []
    for i, (x, y) in enumerate(agents_pos):
        # If the agent is blue and blue movement is not allowed, it stays in place
        if agents_side[i] == 1 and not allow_blue_movement:
            new_positions.append([x, y])
            continue

        # Set target x based on final side, but prioritize moving towards opening
        target_x = (
            0 if abs(y) > barrier_width / 2 else (L if agents_side[i] == 1 else -L)
        )

        # Calculate directional movement towards the target with some noise
        Vx = g * (target_x - x)  # Attractive force towards target side or opening
        Vy = (
            -g * (y / abs(y)) if abs(y) > barrier_width / 2 else 0
        )  # Direct toward opening if misaligned

        # Apply repulsive force from nearby agents
        repulsion = repulsive_force(np.array([x, y]), agents_pos)

        # Apply forces and noise to get new position
        dx = dt * (Vx + repulsion[0]) + sigma * np.random.randn() * np.sqrt(dt)
        dy = dt * (Vy + repulsion[1]) + sigma * np.random.randn() * np.sqrt(dt)

        # Update position with boundary constraints
        new_x = np.clip(x + dx, -L, L)
        new_y = np.clip(y + dy, -L, L)
        new_positions.append([new_x, new_y])

    return np.array(new_positions)


# Visualization setup
fig, ax = plt.subplots()
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)

# Set colors based on side: blue for left group (going right), red for right group (going left)
scat = ax.scatter(
    agents_pos[:, 0],
    agents_pos[:, 1],
    c=["blue" if side == 1 else "red" for side in agents_side],
    s=agent_sizes,  # Use random sizes for each agent
)


# Animation update function
def animate(n):
    global agents_pos
    global all_blue_crossed
    # Check if all red agents have crossed the barrier further left than x = -0.1
    all_red_crossed = np.all(agents_pos[agents_side == -1, 0] < -blue_distance)

    # Update positions; blue agents only move if all red agents have crossed further left
    agents_pos = update_agents(
        agents_pos, agents_side, allow_blue_movement=all_red_crossed
    )
    scat.set_offsets(agents_pos)
    ax.set_title(f"Time = {n * dt:.2f}")

    # Check if all agents have crossed to their respective sides
    if all_blue_crossed is False:
        all_blue_crossed = np.all(agents_pos[agents_side == 1, 0] > 0)
    if all_red_crossed and all_blue_crossed:
        ani.event_source.stop()  # Stop the animation
        ax.set_title(f"Final Time = {n * dt:.2f}")  # Display final time


# Draw barrier with a single opening in the center
ax.plot(
    [-0.05, -0.05], [-L, -barrier_width / 2], color="black", linewidth=2
)  # Left side of the barrier
ax.plot(
    [-0.05, -0.05], [barrier_width / 2, L], color="black", linewidth=2
)  # Right side of the barrier

# Create and run the animation
ani = animation.FuncAnimation(fig, animate, frames=int(T / dt), interval=50)
plt.show()


@dataclasses.dataclass
class Coord:
    def __init__(
        self,
        x,
        y,
    ):
        self._x = x
        self._y = y


@dataclasses.dataclass
class Obstacle:
    def __init__(
        self,
        coord,
    ):
        self._coord = coord


@dataclasses.dataclass
class Wall:
    def __init__(
        self,
        coord,
        color,
    ):
        self._coord = coord
        self._color = color


@dataclasses.dataclass
class Simulation:
    def __init__(
        self,
        agent_list,
        map,
        timer,
        simulation_timer_condition,
    ):
        self._agent_list = agent_list
        self._map = map
        self._timer = timer
        self._simulation_timer_condition = simulation_timer_condition
