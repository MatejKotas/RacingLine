import json
import torch
from track import Track

import matplotlib.pyplot as plt

class TrackFile:
    def __init__(self, name, left, right, resolution, cyclic): # left and right are the filepaths to the json containing the track data
        self.name = name
        self.left = left
        self.right = right
        self.resolution = resolution
        self.cyclic = cyclic

    # Converts bezier cuves from the json files to point lists
    def to_track(self):
        with open(self.left, "r") as f:
            left_data = json.load(f)
        with open(self.right, "r") as f:
            right_data = json.load(f)

        left_points = self.interpolate_bezier(left_data)
        right_points = self.interpolate_bezier(right_data)

        return Track(self.name, left_points, right_points)
    
    def interpolate_bezier(self, data_in):
        length = len(data_in)

        if not self.cyclic:
            length -= 1

        data_out = torch.zeros((length, self.resolution, 2))
        i = 0

        t = torch.arange(0, 1, 1/(self.resolution)).reshape(-1, 1)
        f0 = (1 - t)**3
        f1 = 3 * t * (1 - t)**2
        f2 = 3 * (1 - t) * t**2
        f3 = t**3
        
        for i in range(length):
            _from = data_in[i]
            _to = data_in[(i + 1) % len(data_in)]
            x0 = torch.tensor([_from["px"], -_from["py"]])
            x1 = torch.tensor([_from["hrx"], -_from["hry"]])
            x2 = torch.tensor([_to["hlx"], -_to["hly"]])
            x3 = torch.tensor([_to["px"], -_to["py"]])

            data_out[i] = (x0 * f0) + (x1 * f1) + (x2 * f2) + (x3 * f3)

        return data_out.reshape(-1, 2)

TrackCollection = [
        TrackFile(
            "WeatherTech Raceway Laguna Seca", 
            "tracks/laguna_seca/left.json", 
            "tracks/laguna_seca/right.json",
            3, # Was 48
            True),
        TrackFile(
            "Straight",
            "tracks/straight/left.json",
            "tracks/straight/right.json",
            10,
            False
        ),
    ]
