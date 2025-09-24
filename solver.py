import matplotlib.pyplot as plt
import math
import torch

from car import Car

POSITIONS = 7
FWD_VELOCITIES = 64
SIDE_VELOCITIES = 15
MAX_FWD_VELOCITY = 50  # meters per second
MAX_SIDE_VELOCITY = 10 # meters per second

MAX_ANGLE = 10
MAX_DOT = math.cos(20 / 360 * 2 * math.pi)

def solve(solving_track, car: Car, start="right", cutoff=-1):
    length = solving_track.left_points.shape[0] # The number of checkpoints. The number of checkpoint transitions is length - 1
    
    if cutoff > 0:
        length = cutoff
    
    generator = torch.linspace(0, 1, POSITIONS).reshape(1, -1, 1)

    positions = ((solving_track.left_points.reshape(-1, 1, 2) * (1 - generator)) + (
        solving_track.right_points.reshape(-1, 1, 2) * generator
    ))[:length, ...]
    assert positions.shape == (length, POSITIONS, 2), positions.shape

    generator = None

    side_facing = (solving_track.right_points - solving_track.left_points)[:length]
    side_facing = side_facing / torch.linalg.norm(side_facing, axis=1, keepdims=True)
    fwd_facing = side_facing @ torch.Tensor([[0, 1], [-1, 0]])

    fwd_velocities = fwd_facing.reshape(length, 1, 2) * torch.linspace(MAX_FWD_VELOCITY / FWD_VELOCITIES, MAX_FWD_VELOCITY, FWD_VELOCITIES).reshape(FWD_VELOCITIES, 1)
    side_velocities = side_facing.reshape(length, 1, 2) * torch.linspace(-MAX_SIDE_VELOCITY, MAX_SIDE_VELOCITY, SIDE_VELOCITIES).reshape(SIDE_VELOCITIES, 1)

    velocities = fwd_velocities.reshape(length, FWD_VELOCITIES, 1, 2) + side_velocities.reshape(length, 1, SIDE_VELOCITIES, 2)
    assert velocities.shape == (length, FWD_VELOCITIES, SIDE_VELOCITIES, 2), fwd_velocities.shape

    fwd_facing = None
    side_facing = None
    fwd_velocities = None
    side_velocities = None

    costs = torch.zeros((length - 1, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES))

    physics_penalty = float("inf")

    for i in range(length - 1):
        print(f"Evaluating step { i + 1 } of { length - 1 } ({ i / (length - 1) * 100 }%)")
        delta_pos = positions[i + 1].reshape(POSITIONS, 2) - positions[i].reshape(POSITIONS, 1, 2)
        delta_v = velocities[i + 1].reshape(FWD_VELOCITIES, SIDE_VELOCITIES, 2) - velocities[i].reshape(FWD_VELOCITIES, SIDE_VELOCITIES, 1, 1, 2)
        v_avg = (velocities[i + 1].reshape(FWD_VELOCITIES, SIDE_VELOCITIES, 2) + velocities[i].reshape(FWD_VELOCITIES, SIDE_VELOCITIES, 1, 1, 2)) / 2

        delta_pos = delta_pos.reshape(POSITIONS, 1, 1, POSITIONS, 1, 1, 2)
        delta_v = delta_v.reshape(1, FWD_VELOCITIES, SIDE_VELOCITIES, 1, FWD_VELOCITIES, SIDE_VELOCITIES, 2)
        v_avg = v_avg.reshape(1, FWD_VELOCITIES, SIDE_VELOCITIES, 1, FWD_VELOCITIES, SIDE_VELOCITIES, 2)

        # Time is cost to begin with, physics constraints are added after
        delta_pos_l = torch.linalg.norm(delta_pos, axis=6, keepdims=True)
        v_avg_l = torch.linalg.norm(v_avg, axis=6, keepdims=True)
        v_avg_norm = v_avg / v_avg_l
        dot = ((delta_pos / delta_pos_l) * v_avg_norm).sum(axis=6, keepdims=True)

        cost = delta_pos_l / v_avg_l / dot
        cost = cost.squeeze(axis=6)
        dot = dot.squeeze(axis=6)

        assert cost.shape == (POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES), cost.shape

        a_avg = delta_v / cost[..., None]
        cost = torch.where(dot > MAX_DOT, cost, physics_penalty)

        #### Physics Engine ####

        # Apply cornering constraint
        a_avg_l = torch.linalg.norm(a_avg, axis=6)
        cost = torch.where(a_avg_l <= car.cornering, cost, physics_penalty)

        # Apply braking constraint
        tangential_a = (a_avg * v_avg_norm).sum(axis=6)
        cost = torch.where(tangential_a >= -car.braking, cost, physics_penalty)

        # Apply acceleration constraint
        cost = torch.where(tangential_a <= car.acceleration, cost, physics_penalty)

        costs[i] = cost

    print("Finishing touches...")
    # Find the best path

    costs = costs.reshape(length - 1, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES, -1) # Per edge
    gradients = torch.zeros((length - 1, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES), dtype=torch.int) # Edge indexes
    accumulated_costs = torch.zeros((length, POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES)) # Per node

    for i in range(length - 2, -1, -1):
        full_cost = costs[i] + accumulated_costs[i + 1].reshape(-1)

        accumulated_costs[i], gradients[i] = torch.min(full_cost, axis=3)
    
    costs = None
    
    path = torch.zeros((length, 2))
    path_indexes = torch.zeros((length), dtype=torch.int)
    path_velocities = torch.zeros((length, 2))
    path_velocity_indexes = torch.zeros((length, 2), dtype=torch.int)

    if start == "left":
        current_pos = 0

    elif start == "right":
        current_pos = POSITIONS - 1

    elif start == "center":
        current_pos = POSITIONS // 2

    start_pos = current_pos
    start_fwd_v = current_fwd_v = 0
    start_side_v = current_side_v = SIDE_VELOCITIES // 2

    path[0] = positions[0, current_pos]
    path_indexes[0] = current_pos
    path_velocities[0] = velocities[0, current_fwd_v, current_side_v]
    path_velocity_indexes[0, 0] = current_fwd_v
    path_velocity_indexes[0, 1] = current_side_v

    for i in range(length - 1):
        current_pos, current_fwd_v, current_side_v = torch.unravel_index(gradients[i, current_pos, current_fwd_v, current_side_v], (POSITIONS, FWD_VELOCITIES, SIDE_VELOCITIES))
        
        path_indexes[i + 1] = current_pos
        path_velocity_indexes[i + 1, 0] = current_fwd_v
        path_velocity_indexes[i + 1, 1] = current_side_v

        path[i + 1] = positions[i + 1, current_pos]
        path_velocities[i + 1] = velocities[i + 1, current_fwd_v, current_side_v]

    assert path.shape == (length, 2), path.shape

    print("Done")

    return path, path_indexes, path_velocities, path_velocity_indexes, accumulated_costs[0, start_pos, start_fwd_v, start_side_v]