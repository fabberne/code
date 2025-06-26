import numpy as np

from Material import *

class Fiber: 

    def __init__(self, coords, nodes=None, mat=None):
        
        self.nodes     = nodes
        self.coords    = coords
        self.y, self.z = self.get_sorted_coordinates()

        self.A           = self.area()
        self.Cy, self.Cz = self.centroid()
        self.Iy, self.Iz = self.local_inertia()

        if mat == "Concrete_C30_37":
            self.material = Concrete_C30_37()
        elif mat == "Steel_S235":
            self.material = Steel_S235()
        elif mat == "Rebar_B500B":
            self.material = Rebar_B500B()
        else:
            self.material = Unknown()


    def get_sorted_coordinates(self):
        Cy_geom = np.mean(self.coords[:, 0])
        Cz_geom = np.mean(self.coords[:, 1])
        
        # Compute angles and sort points counterclockwise
        angles = np.arctan2(self.coords[:, 1] - Cz_geom, self.coords[:, 0] - Cy_geom)
        sorted_indices = np.argsort(angles)
        sorted_points  = self.coords[sorted_indices]

        # Close the polygon by appending the first point at the end
        sorted_points = np.vstack([sorted_points, sorted_points[0]])

        # Extract sorted x and y coordinates
        y_sorted = sorted_points[:, 0]
        z_sorted = sorted_points[:, 1]

        return y_sorted, z_sorted

    def area(self):
        return 0.5 * np.sum(self.y[:-1] * self.z[1:] - self.y[1:] * self.z[:-1])


    def centroid(self):
        c_y = (1 / (6 * self.A)) * np.sum((self.y[:-1] + self.y[1:]) * 
                                        (self.y[:-1] * self.z[1:] - self.y[1:] * self.z[:-1]))
        c_z = (1 / (6 * self.A)) * np.sum((self.z[:-1] + self.z[1:]) * 
                                        (self.y[:-1] * self.z[1:] - self.y[1:] * self.z[:-1]))

        return c_y, c_z


    def local_inertia(self):
        Iz = (1 / 12) * np.sum((self.y[  :-1] ** 2 + self.y[:-1] * self.y[1:] + self.y[1:] ** 2) *
                               (self.y[  :-1] * self.z[ 1:  ] - 
                                self.y[ 1:  ] * self.z[  :-1]))
        
        

        Iy = (1 / 12) * np.sum((self.z[:-1] ** 2 + self.z[:-1] * self.z[1:] + self.z[1:] ** 2) *
                               (self.y[  :-1] * self.z[ 1:  ] - 
                                self.y[ 1:  ] * self.z[  :-1]))

        Iy  = Iy - self.A * self.Cz ** 2
        Iz  = Iz - self.A * self.Cy ** 2

        return Iy, Iz