# -*- coding: utf-8 -*-
"""
Transformer Encoder + Regression Head for Q-value prediction
"""

import random
import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from numba import njit
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



def generate_maze(size):
    # Ensure odd size
    if size % 2 == 0:
        size += 1

    maze = np.ones((size, size, size), dtype=int)

    stack = [(0, 0, 0)]
    maze[0, 0, 0] = 0
    visited = {(0, 0, 0)}

    directions = [
        (-2, 0, 0), (2, 0, 0),
        (0, -2, 0), (0, 2, 0),
        (0, 0, -2), (0, 0, 2)
    ]

    while stack:
        x, y, z = stack[-1]
        neighbors = []

        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < size and
                0 <= ny < size and
                0 <= nz < size and
                (nx, ny, nz) not in visited):
                neighbors.append((nx, ny, nz))

        if neighbors:
            nx, ny, nz = random.choice(neighbors)

            # Carve the wall between cells
            maze[(x + nx)//2, (y + ny)//2, (z + nz)//2] = 0
            maze[nx, ny, nz] = 0

            visited.add((nx, ny, nz))
            stack.append((nx, ny, nz))
        else:
            stack.pop()

    start = (0, 0, 0)
    goal  = (size - 1, size - 1, size - 1)
    maze[goal] = 0
    """
    # Pad with solid border
    maze = np.pad(maze, pad_width=1, mode='constant', constant_values=1)

    start = (start[0]+1, start[1]+1, start[2]+1)
    goal  = (goal[0]+1, goal[1]+1, goal[2]+1)
    """
    # -------- BFS to compute shortest path --------
    directions = [
        (1,0,0), (-1,0,0),
        (0,1,0), (0,-1,0),
        (0,0,1), (0,0,-1)
    ]

    q = deque([(start, 0)])
    visited = {start}
    path_length = None

    while q:
        (x,y,z), d = q.popleft()

        if (x,y,z) == goal:
            path_length = d
            break

        for dx,dy,dz in directions:
            nx, ny, nz = x+dx, y+dy, z+dz

            if (0 <= nx < maze.shape[0] and
                0 <= ny < maze.shape[1] and
                0 <= nz < maze.shape[2] and
                maze[nx,ny,nz] == 0 and
                (nx,ny,nz) not in visited):

                visited.add((nx,ny,nz))
                q.append(((nx,ny,nz), d+1))
    
    print("Start:", start)
    print("Goal :", goal)
    print("Shortest Path Length:", path_length)
    
    return maze, start, goal

# =========================
# VISUALISATION
# =========================



# =========================
# ENVIRONMENT STEP
# =========================



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




"""
@njit
def q_bellman_sweep(Q,
    maze,
    goal_i,
    goal_j,goal_k,
    episodes,
    alpha,
    gamma,
    size,
    R,C
):
    n_states = size * size *size
    ACTIONS = 6
    rewards = np.zeros(episodes)
    for episode in range(episodes):

        delta = 0.0
        #total_reward = -0.1

        for state in range(n_states):

            for action in range(ACTIONS):

                new_state, reward, terminated = step_jit(
                    state, action, maze, goal_i, goal_j, goal_k,size
                )

                # Using R-Contamination model

                Q[state, action] += alpha * (reward+ gamma*(R*min_local_cube(Q, new_state, size, C)
                                                             + (1.0-R) * np.max(Q[new_state]))
                                          - Q[state, action])
                #state=new_state     


                """

@njit
def run_policy_check(Q, maze, start, goal, max_steps=750):
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


@njit
def q_bellman_sweep(
    Q, maze, start,goal_i, goal_j, goal_k,
    alpha, gamma, size, R, C,
    tol=1e-2, max_sweeps=1000
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

# =========================================================
# Q-VALUE TRANSFORMER MODEL
# =========================================================

class QValueTransformer(nn.Module):
    
    def __init__(self, model_name,n_states):
        self.n_states=n_states
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.reg_head = nn.Linear(hidden_size, n_states*6+2)
        nn.init.uniform_(self.reg_head.weight, a=-1, b=1)
        #nn.init.constant_(self.reg_head.bias, -50)  # approximate mean Q-value


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        #pooled = outputs.last_hidden_state[:, 0, :]
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        q_values = self.reg_head(pooled)
        #q_values = q_values.view(-1, self.n_states, 6)
        loss = None
        if labels is not None:
            loss = nn.functional.smooth_l1_loss(q_values, labels)
            #loss = nn.functional.mse_loss(q_values, labels)


        return {"loss": loss, "logits": q_values}


# =========================================================
# LOAD DATA
# =========================================================

data_folder = "/home/sumanb2/QLearning_3D/data"

file_path = os.path.join(data_folder, "Q_Learn_3D_3000mazes_45000samples.jsonl")

full_dataset = load_dataset("json", data_files=file_path)["train"]

#full_dataset = concatenate_datasets(datasets_list)
print("Total samples:", len(full_dataset))

split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# =========================================================
# TOKENIZER
# =========================================================

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples, n_states):
    tokenized = tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    batch_labels = []

    for completion in examples["completion"]:
        lines = [
            line.strip()
            for line in completion.splitlines()
            if line.strip()
        ]

        # Expected format:
        # R = 0.25
        # C = 2
        # Q Table:
        # ...six Q-values per row...
        R = float(lines[0].split("=", 1)[1])
        C = float(lines[1].split("=", 1)[1])

        if lines[2] != "Q Table:":
            raise ValueError("Expected 'Q Table:' after R and C")

        q_values = []
        for row in lines[3:]:
            values = [float(x) for x in row.split()]
            if len(values) != 6:
                raise ValueError("Each Q-table row must contain 6 values")
            q_values.extend(values)

        if len(q_values) != n_states * 6:
            raise ValueError(
                f"Expected {n_states * 6} Q-values, got {len(q_values)}"
            )

        batch_labels.append([R, C] + q_values)

    tokenized["labels"] = batch_labels
    return tokenized


print("Number of GPUs:", torch.cuda.device_count())


"""
tokenized_train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

#tokenized_val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)
"""

print("Tokenization complete.")

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
# =========================================================
# MODEL INIT
# =========================================================

learning_rate_a = 0.9
discount_factor_g = 0.7
epsilon = 1.0
epsilon_decay = 1/500

SIZE = 5
ACTIONS = 6
EPISODES = int(epsilon / epsilon_decay)  
#MAX_STEPS = 1000000
MAX_STEPS = 500
# R-Contamination parameter
R = 0.2
C = 1
print("SIZE", SIZE)
print("True R:",R)
print("True C:",C)
maze, start, goal = generate_maze(SIZE)

SIZE=maze.shape[0]

#print("goal", goal)

n_states = SIZE * SIZE * SIZE
print("Tokenizing...")
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    fn_kwargs={"n_states": n_states},
)
tokenized_val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    fn_kwargs={"n_states": n_states},
)
Q = np.zeros((n_states, ACTIONS))


#start_state = start[0] * SIZE + start[1]
model = QValueTransformer(model_name,n_states=n_states).to(device)


# =========================================================
# CUSTOM TRAINER (MSE LOSS)
# =========================================================

class QTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # labels are already tensors on correct device
        labels = inputs.pop("labels")
        
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss




training_args = TrainingArguments(
    output_dir="/home/sumanb2/QLearning_3D/data/q_transformer_output5x5_RC",
    save_strategy="epoch",
    num_train_epochs=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=50,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    #gradient_clip_val=1.0

)


trainer = QTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
)

print("Starting training...")
trainer.train()
print("Training complete.")
output_dir="/home/sumanb2/QLearning_3D/data/q_transformer_output5x5_RC_2"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# =========================================================
# TEST ON NEW MAZE
# =========================================================

model.eval()

Q, deltas = q_bellman_sweep(
            Q, maze, start, goal[0], goal[1], goal[2],
            learning_rate_a, discount_factor_g, SIZE, R, C
        )


np.set_printoptions(threshold=np.inf)

#print("Actual Q Table:\n",Q)


grid = maze
n=SIZE        
grid_str=maze_to_string(grid)

test_prompt=(
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
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

pred = outputs["logits"].detach().cpu().numpy().reshape(-1)

pred_R = float(pred[0])
pred_C = int(np.clip(np.rint(pred[1]), 1, SIZE - 1))
pred_q = pred[2:].reshape(n_states, 6)

print("Predicted R:", pred_R)
print("Predicted C:", pred_C)
print("Real R:", R)
print("Real C:", C)
#print("Predicted Q-Table:\n", pred_q)

err = Q - pred_q
print("Q-Table Max Error:\n", np.max(err))
