from track_collection import *
import matplotlib.pyplot as plt
import torch
from solver import solve
import car

def graph(data, name):
    plt.clf()
    plt.plot(torch.arange(data.shape[0]), data)
    plt.savefig(f"figures/{ name }.png", dpi=300)
    plt.show()

track = TrackCollection[0].to_track()
path, path_indexes, path_velocities, path_velocity_indexes, time = solve(track, car.base_car, start="right", cutoff=-1)

print(f"Total time: {time}")

track.render(path, rotate=True, save=True)
graph(torch.linalg.norm(path_velocities, axis=1), "vel")

dv = path_velocities[:-1, :] - path_velocities[1:, :]
dp = path[:-1, :] - path[1:, :]
dt = torch.linalg.norm(dp, axis=1) / torch.linalg.norm(path_velocities[:-1, :], axis=1)

graph(torch.linalg.norm(dv, axis=1) / dt, "acc")