import matplotlib.patches as patches
import matplotlib.tri as tri
import matplotlib.pyplot as plt

import numpy as np
from tabulate import tabulate

import element
import geometry

class Mesh:

    def __init__(self, geometry, mesh_type, mesh_size):
        
        self.mesh_type = mesh_type
        self.mesh_size = mesh_size

        elements, node_coords = geometry.generate_mesh(mesh_type, mesh_size)

        self.elements    = elements
        self.node_coords = node_coords

        self.A           = self.get_A_numerical()
        self.Cx, self.Cy = self.get_centroid()
        self.Ix, self.Iy = self.get_I_numerical()


    def get_A_numerical(self):
        total_area = sum(elem.A for elem in self.elements)

        return total_area

    def get_centroid(self):
        c_x = sum(elem.A * elem.Cx for elem in self.elements) / self.A
        c_y = sum(elem.A * elem.Cy for elem in self.elements) / self.A

        return c_x, c_y
    
    
    def get_I_numerical(self):
        Ix = 0
        Iy = 0

        for elem in self.elements:
            Ix += elem.A * (elem.Cy - self.Cy)**2 + elem.Ix
            Iy += elem.A * (elem.Cx - self.Cx)**2 + elem.Iy

        return Ix, Iy


    def print(self):
        Mesh_properties = [("Mesh Type"         , self.mesh_type       ),
                           ("Number of elements", len(self.elements   )),
                           ("Number of nodes"   , len(self.node_coords)),
                           ("Cross Section Area", str.format('{0:.2f}', self.A))]

        print(tabulate(Mesh_properties, 
                       tablefmt = "fancy_grid", 
                       floatfmt =        ".2f"
             ))

        CS_properties = [(" ", "y", "z")]
    
        CS_properties.append(("Centroid"         , self.Cx, self.Cy))
        CS_properties.append(("Moment of inertia", self.Ix, self.Iy))
    
        print(tabulate(CS_properties, 
                       headers  =   "firstrow", 
                       tablefmt = "fancy_grid", 
                       floatfmt =        ".2f"
             ))


    def plot(self):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        
        fig, ax = plt.subplots(figsize=(6, 6))

        for i, element in enumerate(self.elements):
            x = element.coords[:, 0]
            y = element.coords[:, 1]
            poly = patches.Polygon(np.column_stack([x, y]), 
                                    edgecolor = element.material.color,
                                    facecolor = element.material.color, 
                                    lw        = 0.3,
                                    )
            ax.add_patch(poly)


        ax.scatter(self.node_coords[:, 0], 
                   self.node_coords[:, 1], 
                   c     =   'red', 
                   s     =       2, 
                   label = "Nodes")
        ax.scatter(self.Cx, 
                   self.Cy, 
                   c     =    "blue", 
                   s     =        20, 
                   label = "Centroid")

        ax.set_xlabel("y [$mm$]")
        ax.set_ylabel("z [$mm$]")
        ax.set_frame_on(False)
        ax.set_aspect('equal')
        ax.legend()
        plt.show()
