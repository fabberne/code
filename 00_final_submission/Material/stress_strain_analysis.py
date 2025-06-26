import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, colors

from tabulate import tabulate

from numba import jit
from collections import defaultdict

@jit(nopython=True, cache=True)
def calculate_strains_fast(eps_x, cy_values, cz_values, xsi_y, xsi_z):
    return eps_x + cz_values * xsi_y + cy_values * xsi_z

@jit(nopython=True, cache=True)
def calculate_section_forces_fast(area_values, stresses, cy_values, cz_values):
    N  = np.sum(area_values * stresses) / 1000
    My = np.sum(area_values * stresses * cz_values) / 1e6
    Mz = np.sum(area_values * stresses * cy_values) / 1e6
    return N, My, Mz

class stress_strain_analysis:
    def __init__(self, mesh, Nx=0, My=0, Mz=0):
        self.mesh = mesh
        self.Nx = Nx
        self.My = My
        self.Mz = Mz

        self.eps_x = 0
        self.xsi_y = 0
        self.xsi_z = 0

        self.area_values = np.array([elem.A for elem in self.mesh.elements])
        self.cy_values   = np.array([elem.Cy - self.mesh.Cy for elem in self.mesh.elements])
        self.cz_values   = np.array([elem.Cz - self.mesh.Cz for elem in self.mesh.elements])

        # Create groups for elements with the same material
        self.material_groups = defaultdict(list)
        self.materials = []
        for i, elem in enumerate(self.mesh.elements):
            if elem.material.name not in self.material_groups:
                self.materials.append(elem.material)

            self.material_groups[elem.material.name].append(i)

        # Initialize stress array
        self.strains  = np.zeros(len(self.mesh.elements))
        self.stresses = np.zeros(len(self.mesh.elements))

    def set_strain_and_curvature(self, eps_x, xsi_y, xsi_z):
        self.eps_x = eps_x
        self.xsi_y = xsi_y
        self.xsi_z = xsi_z

    def calculate_strains(self):
        self.strains = calculate_strains_fast(self.eps_x, self.cy_values, self.cz_values, self.xsi_y, self.xsi_z)

    def calculate_stresses(self):
        # Compute stresses for each material group
        for i, (material_name, indices) in enumerate(self.material_groups.items()):
            indices = np.array(indices)  # Faster indexing
            grouped_strains = self.strains[indices]
            
            # Compute stresses efficiently using the vectorized get_stress
            self.stresses[indices] = self.materials[i].get_stress_vectorized(grouped_strains)

    def get_section_forces(self):
        N, My, Mz = calculate_section_forces_fast(self.area_values, self.stresses, self.cy_values, self.cz_values)
        return N, My, Mz
    
    def system_of_equations(self, V, N_target, My_target, Mz_target):
        self.set_strain_and_curvature(V[0], V[1], V[2])
        self.calculate_strains()
        self.calculate_stresses()
        Nx, My, Mz = self.get_section_forces()
        return [Nx - N_target, My - My_target, Mz - Mz_target]