import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Laminate:
    def __init__(self, layers):
        """
        layers: list of dictionaries, each containing:
            - 'E1': Young's modulus in fiber direction
            - 'E2': Young's modulus perpendicular to fiber direction
            - 'G12': Shear modulus
            - 'v12': Poisson's ratio
            - 'theta': fiber orientation in degrees
            - 'thickness': thickness of the ply
        """
        self.layers = layers
        self.n      = len(layers)
        self.h      = [-sum(l['thickness'] for l in layers) / 2]
        for i, layer in enumerate(self.layers):
            self.h.append(self.h[-1] + layer['thickness'])
        self.A, self.B, self.D = self.compute_ABD_matrix()
        self.Ex, self.Ey, self.Gxy, self.vxy = self.compute_equivalent_properties()
    
    def Q_matrix(self, E1, E2, G12, v12):
        """Computes the reduced stiffness matrix Q for a single ply"""
        v21 = (E2 * v12) / E1  # Reciprocal Poisson's ratio
        Q11 = E1 / (1 - v12 * v21)
        Q22 = E2 / (1 - v12 * v21)
        Q12 = v12 * E2 / (1 - v12 * v21)
        Q66 = G12
        return np.array([[Q11, Q12,    0],
                         [Q12, Q22,    0],
                         [  0,   0,  Q66]])
    
    def transform_Q(self, Q, theta):
        """Transforms the stiffness matrix Q to the laminate coordinate system"""
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        T = np.array([[ c**2, s**2,       2*c*s],
                      [ s**2, c**2,      -2*c*s],
                      [-c*s ,  c*s, c**2 - s**2]])
        return T @ Q @ np.linalg.inv(T)
    
    def compute_ABD_matrix(self):
        """Computes the A, B, and D matrices of the laminate"""
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        for i, layer in enumerate(self.layers):
            Q = self.Q_matrix(layer['E1'], layer['E2'], layer['G12'], layer['v12'])
            Q_bar = self.transform_Q(Q, layer['theta'])
            h_k = self.h[i]
            h_k1 = self.h[i + 1]
            A += Q_bar * (h_k1 - h_k)
            B += (1/2) * Q_bar * (h_k1**2 - h_k**2)
            D += (1/3) * Q_bar * (h_k1**3 - h_k**3)
        
        return A, B, D
    
    def compute_equivalent_properties(self):
        """Computes the equivalent material properties of the laminate."""
        h_total = self.h[-1] - self.h[0]
        Ex  = self.A[0, 0] / h_total
        Ey  = self.A[1, 1] / h_total
        Gxy = self.A[2, 2] / h_total
        vxy = self.A[0, 1] / self.A[1, 1]
        return Ex, Ey, Gxy, vxy

    def get_ABD_matrix(self):
        return self.A, self.B, self.D
    
    def get_equivalent_properties(self):
        return self.Ex, self.Ey, self.Gxy, self.vxy

    def print_results(self):
        """Prints the ABD matrix and equivalent properties in tabular format."""
        headers = ["A Matrix", "B Matrix", "D Matrix"]
        data = [[np.array2string(self.A, formatter={'float_kind': lambda x: f'{x:.2e}'}),
                np.array2string(self.B, formatter={'float_kind': lambda x: f'{x:.2e}'}),
                np.array2string(self.D, formatter={'float_kind': lambda x: f'{x:.2e}'})]]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

        properties = [
            ["Ex" , f"{self.Ex:.2e}"],
            ["Ey" , f"{self.Ey:.2e}"],
            ["Gxy", f"{self.Gxy:.2e}"],
            ["vxy", f"{self.vxy:.2e}"]
        ]
        print(tabulate(properties, headers=["Property", "Value"], tablefmt="fancy_grid"))


    def plot(self):
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
        names  = []

        total_height = sum(l['thickness'] for l in self.layers)
        current_z = 0
        width = total_height * 5  # Base dimensions for the plot
        lengths = [width + (len(self.layers) - i - 1) * 2 * total_height for i in range(len(self.layers))]


        for i, layer in enumerate(self.layers):
            if layer["name"] in names:
                name  = layer["name"]
                color = colors[names.index(name)]
            else:
                names.append(layer["name"])
                name  = layer["name"]
                color = colors[len(names) - 1]

            # Define vertices of the ply
            vertices = [
                [         0,     0, current_z],
                [lengths[i],     0, current_z],
                [lengths[i], width, current_z],
                [         0, width, current_z],
                [         0,     0, current_z + layer['thickness']],
                [lengths[i],     0, current_z + layer['thickness']],
                [lengths[i], width, current_z + layer['thickness']],
                [         0, width, current_z + layer['thickness']]
            ]
            
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side 1
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # Side 2
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Side 3
                [vertices[3], vertices[0], vertices[4], vertices[7]]   # Side 4
            ]

            ax.add_collection3d(Poly3DCollection(faces, color=color, edgecolor='black', zorder=i))

            # Fiber orientation lines
            num_lines = width / total_height * 2
            for j in range(int(2*num_lines)):
                alpha = np.radians(layer['theta'])

                if alpha != np.pi / 2:
                    spacing = width / num_lines / np.cos(alpha)
                    x_start = 0
                    y_start = spacing * j if alpha >= 0 else width - spacing * j

                    x_end = lengths[i]
                    y_end = y_start + lengths[i] * np.tan(alpha)
                    if y_end > width and alpha != 0:
                        y_end = width
                        x_end = (width - y_start) / np.tan(alpha)
                    if y_end < 0 and alpha != 0:
                        y_end = 0
                        x_end = - (y_start) / np.tan(alpha)

                    if y_start <= width and y_start >= 0:
                        ax.plot([x_start, x_end],
                                [y_start, y_end],
                                [current_z + layer['thickness']], 
                                color='k', lw=1, zorder=i)
                
                if alpha != 0:
                    spacing = abs(width / num_lines / np.sin(alpha))
                    x_start = spacing * (j + 1)
                    y_start = 0 if alpha > 0 else width

                    x_end = x_start + abs(width / np.tan(alpha))
                    y_end = width if alpha > 0 else 0

                    if x_end > lengths[i]:
                        y_end = y_start + (lengths[i] - x_start) * np.tan(alpha)
                        x_end = lengths[i]

                    if x_start < lengths[i]:
                        ax.plot([x_start, x_end],
                                [y_start, y_end],
                                [current_z + layer['thickness']], 
                                color='k', lw=1, zorder=i)
                
            # Move to next layer
            current_z += layer['thickness']

        for i in enumerate(names):
            legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=names[i]) for i, name in enumerate(names)]

        # Formatting the 3D plot
        ax.set_xlim([0, lengths[0] + total_height])
        ax.set_ylim([-total_height, width + total_height])
        ax.set_zlim([-total_height, current_z + total_height])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_aspect('equal', "box")
        ax.legend(handles=legend_elements)
        plt.show()



class LaminateLoadAnalysis:
    def __init__(self, laminate):
        self.laminate = laminate
    

    def apply_load(self, Nx, Ny, Nxy, Mx=0, My=0, Mxy=0):
        self.N = np.array([Nx, Ny, Nxy])
        self.M = np.array([Mx, My, Mxy])
        
        ABD = np.block([[self.laminate.A, self.laminate.B], [self.laminate.B, self.laminate.D]])
        
        loads = np.concatenate((self.N, self.M))
        
        midplane_strains_curvatures = np.linalg.solve(ABD, loads)
        midplane_strains = midplane_strains_curvatures[:3]
        midplane_curvatures = midplane_strains_curvatures[3:]

        self.midplane_strains = midplane_strains
        self.midplane_curvatures = midplane_curvatures
        
        return midplane_strains, midplane_curvatures


    def compute_ply_stresses_strains(self, midplane_strains, midplane_curvatures):
        ply_stresses = []
        ply_strains = []
        
        for i, layer in enumerate(self.laminate.layers):
            Q = self.laminate.Q_matrix(layer['E1'], layer['E2'], layer['G12'], layer['v12'])
            Q_bar = self.laminate.transform_Q(Q, layer['theta'])
            
            z = self.laminate.h[i] + layer["thickness"] / 2   # Abstand zur Mittelfläche
            

            strain = midplane_strains + z * midplane_curvatures
            stress = Q_bar @ strain
            
            ply_strains.append(strain)
            ply_stresses.append(stress)
        
        return np.array(ply_strains), np.array(ply_stresses)

    def compute_stress_strains_for_plot(self, midplane_strains, midplane_curvatures):
        ply_stresses = []
        ply_strains = []
        
        for i, layer in enumerate(self.laminate.layers):
            Q = self.laminate.Q_matrix(layer['E1'], layer['E2'], layer['G12'], layer['v12'])
            Q_bar = self.laminate.transform_Q(Q, layer['theta'])
            
            z_1 = self.laminate.h[i]   # Abstand zur Mittelfläche
            z_2 = self.laminate.h[i] + layer["thickness"]   # Abstand zur Mittelfläche
            

            strain_1 = midplane_strains + z_1 * midplane_curvatures
            stress_1 = Q_bar @ strain_1
            strain_2 = midplane_strains + z_2 * midplane_curvatures
            stress_2 = Q_bar @ strain_2
            
            ply_strains.append(strain_1)
            ply_strains.append(strain_2)
            ply_stresses.append(stress_1)
            ply_stresses.append(stress_2)
        
        return np.array(ply_strains), np.array(ply_stresses)


    def print_ply_results(self, ply_strains, ply_stresses):
        strain_table = [[i + 1] + list(map(lambda x: f"{x:.2e}", ply_strains[i])) for i in range(len(ply_strains))]
        stress_table = [[i + 1] + list(map(lambda x: f"{x:.2e}", ply_stresses[i])) for i in range(len(ply_stresses))]
        
        print("\nstrains per layer:")
        print(tabulate(strain_table, headers=["layer", "e_xx", "e_yy", "e_xy"], tablefmt="fancy_grid"))
        
        print("\nstresses per layer:")
        print(tabulate(stress_table, headers=["layer", "s_xx", "s_yy", "t_xy"], tablefmt="fancy_grid"))


    def plot_stress_strain_variation(self):
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        fig, ax = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 3, 3]}, figsize=(8, 4), sharey=True)

        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
        names  = []

        total_thickness = sum(layer['thickness'] for layer in self.laminate.layers)
    
        z_positions = []
        for i, l in enumerate(self.laminate.layers):
            z_positions.append(self.laminate.h[i])
            z_positions.append(self.laminate.h[i] + l["thickness"])

        # Laminate plot
        current_height = -total_thickness/2
        for i, layer in enumerate(self.laminate.layers):
            if layer["name"] in names:
                name  = layer["name"]
                color = colors[names.index(name)]
            else:
                names.append(layer["name"])
                name  = layer["name"]
                color = colors[len(names) - 1]

            ax[0].add_patch(plt.Rectangle((0, current_height), 1, layer['thickness'],
                                          facecolor=color, edgecolor='black'))
            current_height += layer['thickness']

        ax[0].set_xticks([])
        ax[0].set_ylim(-total_thickness/2, total_thickness/2)
        ax[0].set_yticks(z_positions)
        ax[0].set_ylabel(r"Height [$mm$]")

        for i in enumerate(names):
            legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=names[i]) for i, name in enumerate(names)]
        ax[0].legend(handles=legend_elements, loc='upper right')


        strains, stresses = self.compute_stress_strains_for_plot(self.midplane_strains, self.midplane_curvatures)
        # Strain variation plot
        strain_x = np.array([strain[0] for strain in strains])
        ax[1].plot(strain_x, z_positions, color='C0', label=r"$\varepsilon_{xx}$")
        strain_y = np.array([strain[1] for strain in strains])
        ax[1].plot(strain_y, z_positions, color='C1', label=r"$\varepsilon_{yy}$")
        ax[1].legend(loc='upper right')
        ax[1].invert_yaxis()
        ax[1].set_xlabel(r"Strain [$\frac{mm}{mm}$]")

        # Stress variation plot
        stress_x = np.array([stress[0] for stress in stresses])
        ax[2].plot(stress_x, z_positions, color='C0', label=r"$\sigma_{xx}$")
        stress_y = np.array([stress[1] for stress in stresses])
        ax[2].plot(stress_y, z_positions, color='C1', label=r"$\sigma_{yy}$")
        ax[2].legend(loc='upper right')
        ax[2].invert_yaxis()
        ax[2].set_xlabel(r"Stress [$N/mm^2$]")

        for a in ax:
            a.axhline(0, color='black', linestyle='--', lw=0.5)
            a.axvline(0, color='black', linestyle='-', lw=0.5)
            for i, h in enumerate(self.laminate.h):
                a.axhline(h,color='k', linestyle='-', lw=0.5, alpha=0.5)

        plt.suptitle(r"$N_x = {}$, $N_y = {}$, $N_{{xy}} = {}$".format(
                     self.N[0], self.N[1], self.N[2])+
                     "\n"+
                     r"$M_x = {}$, $M_y = {}$, $M_{{xy}} = {}$".format(
                     self.M[0], self.M[1], self.M[2]))
        plt.subplots_adjust(wspace=0.3)

        plt.tight_layout()
        plt.show()