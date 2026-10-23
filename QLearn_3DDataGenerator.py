import random
import matplotlib.pyplot as plt
import json
from ollama import Client
import os
import numpy as np
from numba import njit
from collections import deque
from tqdm import tqdm
# =========================
# MAZE GENERATION
# =========================

def generate_maze(size, num_obstacles=300, max_obstacle_size=1):

    maze = np.zeros((size, size, size), dtype=int)

    start = (0, 0, 0)
    goal = (size - 1, size - 1, size - 1)

    directions = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
    ]

    # ==========================================
    # BFS: Check whether a valid path exists
    # ==========================================

    def bfs_path_exists(maze):

        q = deque([start])
        visited = {start}

        while q:

            x, y, z = q.popleft()

            if (x, y, z) == goal:
                return True

            for dx, dy, dz in directions:

                nx = x + dx
                ny = y + dy
                nz = z + dz

                if (
                    0 <= nx < size
                    and 0 <= ny < size
                    and 0 <= nz < size
                    and maze[nx, ny, nz] == 0
                    and (nx, ny, nz) not in visited
                ):

                    visited.add((nx, ny, nz))
                    q.append((nx, ny, nz))

        return False

    # ==========================================
    # Generate 10 random obstacles
    # ==========================================

    obstacles_placed = 0
    attempts = 0
    max_attempts = 10000

    while obstacles_placed < num_obstacles and attempts < max_attempts:

        attempts += 1

        # Random obstacle dimensions
        dx = random.randint(1, min(max_obstacle_size, size))
        dy = random.randint(1, min(max_obstacle_size, size))
        dz = random.randint(1, min(max_obstacle_size, size))

        # Random obstacle position
        x = random.randint(0, size - dx)
        y = random.randint(0, size - dy)
        z = random.randint(0, size - dz)

        # Avoid overlapping existing obstacles
        if np.any(maze[x:x+dx, y:y+dy, z:z+dz] == 1):
            continue

        # Temporarily place obstacle
        maze[x:x+dx, y:y+dy, z:z+dz] = 1

        # Start and goal must remain free
        if maze[start] == 1 or maze[goal] == 1:
            maze[x:x+dx, y:y+dy, z:z+dz] = 0
            continue

        # Ensure path still exists
        if bfs_path_exists(maze):
            obstacles_placed += 1

        else:
            # Remove obstacle if it blocks all paths
            maze[x:x+dx, y:y+dy, z:z+dz] = 0

    if obstacles_placed < num_obstacles:
        raise RuntimeError(
            f"Could only place {obstacles_placed} of "
            f"{num_obstacles} obstacles while preserving a path."
        )

    #print("Start:", start)
    #print("Goal:", goal)
    #print("Obstacles placed:", obstacles_placed)

    return maze, start, goal
# =========================
# VISUALISATION
# =========================
def run_policy(Q, maze, start, goal, max_steps=1750):
    size = maze.shape[0]

    # Encode start state (3D → 1D)
    state = start[0]*size*size + start[1]*size + start[2]

    goal_state = goal[0]*size*size + goal[1]*size + goal[2]

    print("Start:", start)
    print("Goal :", goal)

    path = [start]

    for _ in range(max_steps):

        # Greedy action
        action = np.argmax(Q[state])

        new_state, _, done = step_jit(
            state, action, maze,
            goal[0], goal[1], goal[2],
            size
        )

        state = new_state

        # Decode state (1D → 3D)
        x = state // (size * size)
        y = (state % (size * size)) // size
        z = state % size

        path.append((x, y, z))

        # Check goal reached
        if done and state == goal_state:
            print("Reached goal in steps:", len(path) - 1)
            return path

    print("Did not reach goal")
    return path

@njit
def run_policy_check(Q, maze, start, goal, max_steps=1750):
    size = maze.shape[0]

    # Encode start state (3D → 1D)
    state = start[0]*size*size + start[1]*size + start[2]

    goal_state = goal[0]*size*size + goal[1]*size + goal[2]

    path = [start]

    for _ in range(max_steps):

        # Greedy action
        action = np.argmax(Q[state])

        new_state, _, done = step_jit(
            state, action, maze,
            goal[0], goal[1], goal[2],
            size
        )

        state = new_state

        # Decode state (1D → 3D)
        x = state // (size * size)
        y = (state % (size * size)) // size
        z = state % size

        path.append((x, y, z))

        # Check goal reached
        if done and state == goal_state:
            #print("Reached goal in steps:", len(path) - 1)
            return True

    #print("Did not reach goal")
    return False

"""
def plot_maze(maze, start, goal):
    img = np.zeros((maze.shape[0], maze.shape[1], 3))
    img[maze == 0] = [1,1,1]
    img[maze == 1] = [0,0,0]
    img[start] = [0,1,0]
    img[goal]  = [1,0,0]

    plt.figure(figsize=(7,7))
    plt.imshow(img, interpolation="nearest")
    plt.axis("off")
    plt.show()
"""
# =========================
# ENVIRONMENT STEP
# =========================
def plot_path_on_maze(maze, path, start, goal):

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # ---- OBSTACLES (BLUE BLOCKS) ----
    walls = maze == 1

    ax.voxels(
        walls,
        facecolors='royalblue',
        edgecolor='navy',
        linewidth=0.5,
        alpha=0.55
    )

    # ---- LEARNED PATH (THICK ORANGE LINE) ----
    xs = [p[0] + 0.5 for p in path]
    ys = [p[1] + 0.5 for p in path]
    zs = [p[2] + 0.5 for p in path]

    ax.plot(
        xs, ys, zs,
        color='darkorange',
        linewidth=4,
        label='Learned Path',
        zorder=10
    )

    ax.scatter(
        xs, ys, zs,
        color='darkorange',
        s=18,
        depthshade=False
    )

    # ---- START AND GOAL ----
    ax.scatter(
        start[0] + 0.5,
        start[1] + 0.5,
        start[2] + 0.5,
        color='limegreen',
        s=180,
        label='Start',
        depthshade=False
    )

    ax.scatter(
        goal[0] + 0.5,
        goal[1] + 0.5,
        goal[2] + 0.5,
        color='red',
        s=180,
        label='Goal',
        depthshade=False
    )

    # ---- AXIS SETTINGS ----
    size = maze.shape[0]

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)

    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Obstacle Environment with Learned Q-Learning Path")

    ax.view_init(elev=25, azim=45)

    ax.legend()
    plt.tight_layout()
    plt.show()


# =========================
# Q-LEARNING
# =========================

@njit
def step_jit(state, action, maze, goal_x, goal_y, goal_z, size):
    # Decode state
    x = state // (size * size)
    y = (state % (size * size)) // size
    z = state % size

    nx, ny, nz = x, y, z

    # Actions: 6-connected grid
    if action == 0:
        nx += 1
    elif action == 1:
        nx -= 1
    elif action == 2:
        ny += 1
    elif action == 3:
        ny -= 1
    elif action == 4:
        nz += 1
    else:
        nz -= 1

    # Boundary check
    if (nx < 0 or nx >= size or
        ny < 0 or ny >= size or
        nz < 0 or nz >= size):
        return state, -100, False

    # Obstacle
    if maze[nx, ny, nz] == 1:
        return state, -1.0, False

    # Goal
    if nx == goal_x and ny == goal_y and nz == goal_z:
        new_state = nx * size * size + ny * size + nz
        return new_state, 5.0, True

    # Free space
    new_state = nx * size * size + ny * size + nz
    return new_state, -0.1, False
"""
@njit
def minmax(Q):
    n_states = Q.shape[0]
    n_actions = Q.shape[1]
    
    min_val = 1e18  # large number
    
    for s in range(n_states):
        row_max = Q[s, 0]
        for a in range(1, n_actions):
            if Q[s, a] > row_max:
                row_max = Q[s, a]
        
        if row_max < min_val:
            min_val = row_max
            
    return min_val
    """
"""
@njit
def q_learning_jit(
    Q,
    maze,
    start_state,
    goal_i,
    goal_j,
    episodes,
    max_steps,
    alpha,
    gamma,
    epsilon,
    epsilon_decay,
    size, R
):
    rewards = np.zeros(episodes)
    
    for episode in range(episodes):
        state = start_state
        total_reward = -1.0

        for _ in range(max_steps):
            if np.random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(Q[state])

            new_state, reward, terminated = step_jit(
                state, action, maze, goal_i, goal_j, size
            )
            

            # Using R-Contamination model

            r_min=-100
            contaminated_reward = (1.0 - R) * reward + R * r_min

            Q[state, action] += alpha * (
                    contaminated_reward
                    + gamma * np.max(Q[new_state])
                    - Q[state, action]
                )


            state = new_state
            total_reward += reward

            if terminated:
                
                break

        
        epsilon = min(epsilon - epsilon_decay, 0.01)
        #epsilon = max(epsilon - epsilon_decay, 0.0)
        rewards[episode] = total_reward

    return Q, rewards
"""
@njit
def min_local_cube(Q, state, size, C):
    # Decode state
    x = state // (size * size)
    y = (state % (size * size)) // size
    z = state % size

    worst_val = 1e18  # large number

    for dx in range(-C, C+1):
        for dy in range(-C, C+1):
            for dz in range(-C, C+1):

                nx = x + dx
                ny = y + dy
                nz = z + dz

                # bounds check
                if (0 <= nx < size and
                    0 <= ny < size and
                    0 <= nz < size):

                    neighbor_state = nx * size * size + ny * size + nz

                    # max over actions at this neighbor
                    row_max = Q[neighbor_state, 0]
                    for a in range(1, Q.shape[1]):
                        if Q[neighbor_state, a] > row_max:
                            row_max = Q[neighbor_state, a]

                    if row_max < worst_val:
                        worst_val = row_max

    return worst_val

@njit
def greedy_reaches_goal(Q, maze, start_state,
                        goal_i, goal_j, goal_k, size):
    goal_state = goal_i * size * size + goal_j * size + goal_k
    state = start_state
    visited = np.zeros(size ** 3, dtype=np.bool_)

    while state != goal_state:
        # A repeated state means the deterministic greedy policy is looping.
        if visited[state]:
            return False
        visited[state] = True

        action = np.argmax(Q[state])
        state, _, terminated = step_jit(
            state, action, maze, goal_i, goal_j, goal_k, size
        )

        if terminated:
            return state == goal_state

    return True

@njit
def q_bellman_sweep(
    Q, maze, start,goal_i, goal_j, goal_k,
    alpha, gamma, size, R, C,
    tol=1e-2, max_sweeps=5000
):
    n_states = size ** 3
    ACTIONS = 6
    deltas = np.zeros(max_sweeps)
    goal_state = goal_i * size * size + goal_j * size + goal_k
    for sweep in range(max_sweeps):
        Q_old = Q.copy()

        for state in range(n_states):
            for action in range(ACTIONS):
                new_state, reward, terminated = step_jit(
                    state, action, maze,
                    goal_i, goal_j, goal_k, size
                )

                target = reward + gamma * (
                    R * min_local_cube(Q, new_state, size, C)
                    + (1.0 - R) * np.max(Q[new_state])
                )

                Q[state, action] += alpha * (
                    target - Q[state, action]
                )

        # Sup norm of the Q-function difference, over all (s, a)
        delta = np.max(np.abs(Q - Q_old))
        deltas[sweep] = delta

        if delta <= tol:
            if run_policy_check(Q, maze, start, goal, MAX_STEPS):
                return Q, deltas[:sweep + 1]

    return Q, deltas[:sweep + 1]

    


def maze_to_string(maze):
    size = maze.shape[0]
    s = ""

    for z in range(size):
        s += f"Layer z = {z}\n"
        for x in range(size):
            for y in range(size):
                s += str(maze[x, y, z]) + " "
            s += "\n"
        s += "\n"

    return s
# =========================
# MAIN
# =========================
np.set_printoptions(threshold=np.inf)

if __name__ == "__main__":
    learning_rate_a = 0.9
    #learning_rate_a = 0.9
    discount_factor_g = 0.7
    epsilon = 1.0
    epsilon_decay = 1/500

    SIZE = 11
    ACTIONS = 6
    EPISODES = int(epsilon / epsilon_decay)  
    #MAX_STEPS = 1000000
    MAX_STEPS = 500
    # R-Contamination parameter
    R = 0.2
    p = SIZE
    n=p
    

    client = Client()


    R_VALUES = [0.0, 0.1, 0.2, 0.4, 0.5]
    C_VALUES = [0, 1, 2]

    itr = 2500 # number of iterations
    dataset = []

    SIZE = p
    n = SIZE
    n_states = SIZE ** 3

    folder_path = "/home/sumanb2/QLearning_3D/data"
    os.makedirs(folder_path, exist_ok=True)

    #filename = os.path.join(folder_path,"Q_Learn_3D_3000mazes_45000samples.jsonl")

    filename = os.path.join(folder_path,"Q_Learn_3D_11x11x11maze.jsonl")
    
    # Generate 3000 different mazes
    for k in tqdm(range(itr), desc="Generating mazes"):

        maze, start, goal = generate_maze(SIZE)

        n = maze.shape[0]
        n_states = n ** 3

        # Convert maze to string once per maze
        grid_str = maze_to_string(maze)

        # Generate 15 samples for each maze
        for R in R_VALUES:
            for C in C_VALUES:

                # Initialize a new Q-table
                Q = np.zeros((n_states, ACTIONS))

                # Run robust Q-learning for this (R, C)
                Q, deltas = q_bellman_sweep(
                    Q,
                    maze,
                    start,
                    goal[0],
                    goal[1],
                    goal[2],
                    learning_rate_a,
                    discount_factor_g,
                    n,
                    R,
                    C
                )
                #path=run_policy(Q, maze, start, goal)
                #print("Path:",path)
                #plot_path_on_maze(maze, path, start, goal)
                # Construct the prompt
                prompt = (
                    "### Instruction:\n"
                    f"You are given a {n}x{n}x{n} grid.\n"
                    "The maze is represented as layers along the z-axis.\n"
                    "Each layer is a 2D grid where:\n"
                    "0 = free space, 1 = obstacle.\n\n"
                    f"{grid_str}"
                    f"Start at (0,0,0) and move to ({n-1},{n-1},{n-1})\n"
                    "Output R and C first, followed by the Q Table.\n"
                    "Do NOT include extra text or explanations.\n\n"
                    "### Response:\n"
                )

                # Convert Q-table into string
                q_table_entries = []

                for state in range(Q.shape[0]):
                    q_values_str = " ".join(
                        f"{x:.4f}" for x in Q[state]
                    )
                    q_table_entries.append(q_values_str)

                q_table_str = (
                    f"R = {R}\n"
                    f"C = {C}\n"
                    "Q Table:\n"
                    + "\n".join(q_table_entries)
                )

                entry = {
                    "prompt": prompt.strip(),
                    "completion": q_table_str
                }

                dataset.append(entry)

        # Optional progress print
        if (k + 1) % 100 == 0:
            print(
                f"Generated {k+1} mazes, "
                f"{len(dataset)} samples"
            )

    # Verify total number of samples
    assert len(dataset) == itr * len(R_VALUES) * len(C_VALUES)

    print(f"Total mazes: {itr}")
    print(f"Samples per maze: {len(R_VALUES) * len(C_VALUES)}")
    print(f"Total dataset entries: {len(dataset)}")

    # Save JSONL dataset
    with open(filename, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    print(f"Data saved to {filename}")