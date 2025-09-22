from track_collection import *
# from solver import solve
from car import base_car
import numpy as np

laguna_seca = TrackCollection[0].to_track()
# path = solve(laguna_seca, base_car)

# laguna_seca.render(path, True)

laguna_seca.render(rotate=True)