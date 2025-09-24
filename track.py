import numpy as np
import matplotlib.pyplot as plt
import datetime

# Class to store track data with some utility functions
# left and right must have the same length and be aligned e.g. the first point in left has to be across from the first point in right

class Track:
    def __init__(self, name, left_points, right_points):
        self.name = name
        self.left_points = left_points
        self.right_points = right_points
        assert self.left_points.shape[1] == 2, "Points must have 2 numbers per point"
        assert self.left_points.shape == self.right_points.shape, "Left and right points must have the same shape"

    def render(self, path=None, rotate=False, save=False):
        plt.clf()

        start = np.stack((self.left_points[0], self.right_points[0]))
        
        self.render_helper(start, rotate, "g.-")
        self.render_helper(self.left_points, rotate, "b.-")
        self.render_helper(self.right_points, rotate, "r.-")

        if not path is None:
            self.render_helper(path, rotate, "m.-")

        if save:
            plt.savefig(f"figures/fig.png", dpi=300)
        plt.show()

    def render_helper(self, points, rotate, style):
        if not rotate:
            plt.plot(points.T[0], points.T[1], style, markersize=0, linewidth=1)

        else:
            plt.plot(points.T[1], -points.T[0], style, markersize=0, linewidth=1)