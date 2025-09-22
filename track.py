import numpy as np
import matplotlib.pyplot as plt

# Class to store track data with some utility functions
# left and right must have the same length and be aligned e.g. the first point in left has to be across from the first point in right

class Track:
    def __init__(self, name, left_points, right_points):
        self.name = name
        self.left_points = left_points
        self.right_points = right_points
        assert self.left_points.shape[1] == 2, "Points must have 2 numbers per point"
        assert self.left_points.shape == self.right_points.shape, "Left and right points must have the same shape"

    def render(self, path=None, rotate=False):
        plt.clf()
        
        if not rotate:
            plt.plot(self.left_points.T[0], self.left_points.T[1], "b.-", markersize=0, linewidth=1)
            plt.plot(self.right_points.T[0], self.right_points.T[1], "r.-", markersize=0, linewidth=1)

            if not path is None:
                plt.plot(path.T[0], path.T[1], "r.-", markersize=0, linewidth=1)

        else:
            plt.plot(self.left_points.T[1], -self.left_points.T[0], "b.-", markersize=0, linewidth=1)
            plt.plot(self.right_points.T[1], -self.right_points.T[0], "r.-", markersize=0, linewidth=1)

            if not path is None:
                plt.plot(path.T[1], -path.T[0], "r.-", markersize=0, linewidth=1)

        plt.show()