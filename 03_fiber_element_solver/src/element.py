import numpy as np

import material

class Element_2D:

    def __init__(self, nodes, coords, mat = None):

        self.nodes     = nodes
        self.coords    = coords
        self.x, self.y = self.get_sorted_coordinates()

        self.A           = self.area()
        self.Cx, self.Cy = self.centroid()
        self.Ix, self.Iy = self.local_inertia()

        if mat == "concrete":
            self.material = material.Concrete_C30_37()
        elif mat == "steel":
            self.material = material.Steel_S235()
        elif mat == "rebar":
            self.material = material.Rebar_B500B()
        else:
            self.material = material.Unknown()


    def get_sorted_coordinates(self):
        Cx_geom = np.mean(self.coords[:, 0])
        Cy_geom = np.mean(self.coords[:, 1])
        
        # Compute angles and sort points counterclockwise
        angles = np.arctan2(self.coords[:, 1] - Cy_geom, self.coords[:, 0] - Cx_geom)
        sorted_indices = np.argsort(angles)
        sorted_points  = self.coords[sorted_indices]

        # Close the polygon by appending the first point at the end
        sorted_points = np.vstack([sorted_points, sorted_points[0]])

        # Extract sorted x and y coordinates
        x_sorted = sorted_points[:, 0]
        y_sorted = sorted_points[:, 1]

        return x_sorted, y_sorted

    def area(self):
        return 0.5 * np.sum(self.x[:-1] * self.y[1:] - self.x[1:] * self.y[:-1])


    def centroid(self):
        c_x = (1 / (6 * self.A)) * np.sum((self.x[:-1] + self.x[1:]) * 
                                        (self.x[:-1] * self.y[1:] - self.x[1:] * self.y[:-1]))
        c_y = (1 / (6 * self.A)) * np.sum((self.y[:-1] + self.y[1:]) * 
                                        (self.x[:-1] * self.y[1:] - self.x[1:] * self.y[:-1]))

        return c_x, c_y


    def local_inertia(self):
        Ix = (1 / 12) * np.sum((self.y[  :-1] ** 2 + self.y[:-1] * self.y[1:] + self.y[1:] ** 2) *
                            (self.x[  :-1] * self.y[ 1:  ] - 
                                self.x[ 1:  ] * self.y[  :-1]))

        Iy = (1 / 12) * np.sum((self.x[:-1] ** 2 + self.x[:-1] * self.x[1:] + self.x[1:] ** 2) *
                            (self.x[  :-1] * self.y[ 1:  ] - 
                                self.x[ 1:  ] * self.y[  :-1]))

        Ix  = Ix - self.A * self.Cy ** 2
        Iy  = Iy - self.A * self.Cx ** 2

        return Ix, Iy