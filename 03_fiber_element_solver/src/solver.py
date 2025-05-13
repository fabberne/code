import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from print_color import print
import numpy as np
import pandas as pd
import tqdm

import structure
from constraint import *

class Solver():

    def plot_initial_structure(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for beam in self.structure.beam_elements:
            start_node = beam.nodes_initial[0]
            end_node = beam.nodes_initial[1]

            # Plot nodes
            ax.scatter(*start_node, color='k', marker='o')
            ax.scatter(*end_node, color='k', marker='o')

            # Plot beam line
            ax.plot([start_node[0], end_node[0]],
                    [start_node[1], end_node[1]],
                    [start_node[2], end_node[2]],
                    color='k', linewidth=2)

        ax.set_box_aspect((np.ptp([0,7000]), np.ptp([-2000,2000]), np.ptp([0,9000])))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Frame Structure')
        plt.show()

    def plot_displaced_structure(self, scale=20.0):
        self.structure.set_displaced_nodes(self.displacements, scale)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for beam in self.structure.beam_elements:
            start_node = beam.nodes_initial[0]
            end_node = beam.nodes_initial[1]

            # Plot nodes
            ax.scatter(*start_node, color='k', marker='o', zorder=1)
            ax.scatter(*end_node, color='k', marker='o', zorder=1)

            # Plot beam line
            ax.plot([start_node[0], end_node[0]],
                    [start_node[1], end_node[1]],
                    [start_node[2], end_node[2]],
                    color='k', linewidth=2, zorder=1)

        for beam in self.structure.beam_elements:
            start_node = beam.nodes_displaced[0]
            end_node = beam.nodes_displaced[1]

            # Plot nodes
            ax.scatter(*start_node, color='r', marker='o', zorder=0)
            ax.scatter(*end_node, color='r', marker='o', zorder=0)

            # Plot beam line
            ax.plot([start_node[0], end_node[0]],
                    [start_node[1], end_node[1]],
                    [start_node[2], end_node[2]],
                    color='r', linewidth=2, zorder=0)

        ax.set_ylim([-2000, 2000])
        ax.set_box_aspect((np.ptp([0,7000]), np.ptp([-2000,2000]), np.ptp([0,9000])))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def print_nodal_displacements_and_forces(self):
        # Define labels for the DOFs
        
        dof_labels   = [ 'u [mm]',  'v [mm]',  'w [mm]', 'θx [rad]', 'θy [rad]', 'θz [rad]']
        F_int_labels = ['Fx [kN]', 'Fy [kN]', 'Fz [kN]', 'Mx [kNm]', 'My [kNm]', 'Mz [kNm]']

        for node in range(self.structure.number_of_nodes):
            u_node     = self.displacements[node*6:(node+1)*6].reshape(6)
            F_int_node = self.forces[       node*6:(node+1)*6].reshape(6)

            F_int_node[[0,1,2]] = F_int_node[[0,1,2]] / 1000.0    # Convert to kN
            F_int_node[[3,4,5]] = F_int_node[[3,4,5]] / 1000000.0 # Convert to kNm

            # Create a DataFrame
            df1 = pd.DataFrame({'Displacement'  : u_node    }, index =   dof_labels)
            df2 = pd.DataFrame({'Internal Force': F_int_node}, index = F_int_labels)

            # Format nicely
            pd.set_option('display.precision', 9)
            pd.set_option('display.float_format', '{:,.6f}'.format)
            print("--------------------------------------")
            print("Node", node)
            print(df1)
            print(df2)


#--------------------------------------------------------------------------------------------------------------------------------#


class Linear(Solver):
    def __init__(self, structure):
        self.structure = structure

        self.displacements = np.zeros(self.structure.number_of_DOFs)
        self.forces        = np.zeros(self.structure.number_of_DOFs)

    def solve(self):
        
        self.structure.assemble()
        self.structure.apply_nodal_loads()

        self.displacements = np.linalg.solve(self.structure.K_global, self.structure.F_global)

        self.structure.assemble_without_bc()
        self.forces = np.dot(self.structure.K_global, self.displacements)

        self.print_nodal_displacements_and_forces()

        return
    

#--------------------------------------------------------------------------------------------------------------------------------#


class Nonlinear(Solver):
    def __init__(self, structure, constraint = "Load", 
                 NR_tolerance      = 1e-6, NR_max_iter      = 100, 
                 section_tolerance = 1e-6, section_max_iter = 100, 
                 controlled_DOF=None):

        self.structure = structure

        self.attempts = 5

        self.NR_tolerance = NR_tolerance
        self.NR_max_iter  = NR_max_iter

        self.structure.set_section_max_iter_and_tolerance(section_max_iter, section_tolerance)

        self.iteration = 0
        self.iteration_section = 0

        if constraint == "Load":
            self.constraint = Load()
        elif constraint == "Displacement":
            self.constraint = Displacement()
        elif constraint == "Arc":
            self.constraint = Arc()
        else:
            raise ValueError(f"Unknown constraint type: '{constraint}'. Expected one of 'Load', 'Displacement', or 'Arc'.")
            
        self.controlled_DOF = controlled_DOF


    def solve(self, increments):

        lambda_history = np.zeros(len(increments) + 1)
        u_history      = np.zeros((len(increments) + 1, self.structure.number_of_DOFs))

        for step in tqdm.tqdm_notebook(range(len(increments))):
            print("----------------------------------------------")
            print("Load step", step + 1, "of", len(increments))
            attempt = 1
            convergence_boolean = False

            # displacement and load at the beginning of the step
            u0, lambda0 = self.structure.getState()
            lambda0     = float(lambda0)
            u0          = u0.copy()


            while (not convergence_boolean) and (attempt <= self.attempts):
                print("   Attempt ", attempt)
                attempt += 1

                u, llambda, convergence_boolean = self.getSolution(u0, lambda0, 
                                                                   increments[step], 
                                                                   self.controlled_DOF)

                if (not convergence_boolean) and (attempt <= self.attempts):
                    increments[step] *= 0.5
                    print("   Decreased increment to ", increments[step])

            if not convergence_boolean:
                print("   Failed to reach convergence after ", attempt-1, "attempts")

            u_history[step + 1, :]   = u.reshape(self.structure.number_of_DOFs)
            lambda_history[step + 1] = llambda

        return u_history, lambda_history


    def getSolution(self, u0, lambda0, increment, controlled_DOF=None):

        self.structure.assemble()
        Stiffness_K, fext, ResidualsR = self.structure.K_global, self.structure.F_global, self.structure.Residual
        convergence_norm = max(np.linalg.norm(fext),1)

        #___ Step 1: Predict the solution at the beginning of the step ___
        (u, llambda, deltaUp, deltalambdap, Stiffness_K, fext, ResidualsR,
        ) = self.constraint.predict(self.structure.getSystemMatrices,
                                    u0, lambda0, increment, Stiffness_K, 
                                    fext, ResidualsR)
        
        for iteration in range(self.NR_max_iter):

            #___ Step 2: Calculate the deformation increments ___
            print("      NR Iteration ", iteration)
            g, h, s = self.constraint.get(u, llambda, u0, lambda0, 
                                          deltaUp, deltalambdap,
                                          increment, controlled_DOF=controlled_DOF)

            du_tilde        =   np.linalg.inv(Stiffness_K).dot(fext)
            du_double_tilde = - np.linalg.inv(Stiffness_K).dot(ResidualsR)

            deltalambdap    = - (g + np.transpose(h).dot(du_double_tilde)) / (s +np.transpose(h).dot(du_tilde))
            deltaUp         = deltalambdap * du_tilde + du_double_tilde

            #___ Steps 3 - 14 ___
            Stiffness_K, fext, ResidualsR = self.structure.getSystemMatrices(deltaUp, deltalambdap)
        
            # Update the solution variables
            u       = self.structure.displacements
            llambda = self.structure.lambda_factor_converged
            
            #___ Step 15: Check convergence criteria ___
            print("      Residuals Norm ", np.linalg.norm(ResidualsR))


            if np.linalg.norm(ResidualsR) <= self.NR_tolerance * convergence_norm:
                print("NR Converged!")
                convergence_boolean = True

                # finalize the load step
                self.structure.displacements_increment.fill(0.0)
                self.structure.lambda_factor_increment = 0.0
                self.structure.displacements_converged = self.structure.displacements
                self.structure.lambda_factor_converged = self.structure.lambda_factor
                for beam_element in self.structure.beam_elements:
                    beam_element.resisting_forces_converged = beam_element.resisting_forces
                    beam_element.force_increment.fill(0.0)

                    for section in beam_element.cross_sections:
                        section.forces_converged = section.forces
                        section.forces_increment.fill(0.0)

                        section.strains_converged = section.strains
                        section.strains_increment.fill(0.0)

                break
            else:
                convergence_boolean = False
                
        if self.constraint.name == "Displacement control":
            return -u, -llambda, convergence_boolean
        else:
            return  u,  llambda, convergence_boolean