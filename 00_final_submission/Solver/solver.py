import numpy as np
import pandas as pd
import tqdm

from .constraint import *

class Solver():

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
        section_forces_history  = np.zeros((len(increments), len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        section_strains_history = np.zeros((len(increments), len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))

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

                u, llambda, convergence_boolean, section_forces, section_strains = self.getSolution(u0, lambda0, 
                                                                   increments[step], 
                                                                   self.controlled_DOF)

                if (not convergence_boolean) and (attempt <= self.attempts):
                    increments[step] *= 0.5
                    print("   Decreased increment to ", increments[step])

            if not convergence_boolean:
                print("   Failed to reach convergence after ", attempt-1, "attempts")

            u_history[step + 1, :]   = u.reshape(self.structure.number_of_DOFs)
            lambda_history[step + 1] = llambda
            section_forces_history[step, :, :, :]  = np.array(section_forces)
            section_strains_history[step, :, :, :] = np.array(section_strains)


        return u_history, lambda_history, section_forces_history, section_strains_history


    def getSolution(self, u0, lambda0, increment, controlled_DOF=None):

        self.structure.assemble()
        Stiffness_K, fext, ResidualsR = self.structure.K_global, self.structure.F_global, self.structure.Residual
        convergence_norm = max(np.linalg.norm(fext),1)

        #___ Step 1: Predict the solution at the beginning of the step ___
        (u, llambda, deltaUp, deltalambdap, Stiffness_K, fext, ResidualsR,
        ) = self.constraint.predict(self.structure.getSystemMatrices,
                                    u0, lambda0, increment, Stiffness_K, 
                                    fext, ResidualsR)

        section_forces  = np.zeros((len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        section_strains = np.zeros((len(self.structure.beam_elements), len(self.structure.beam_elements[0].cross_sections), 3))
        
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
                for i, beam_element in enumerate(self.structure.beam_elements):
                    beam_element.resisting_forces_converged = beam_element.resisting_forces
                    beam_element.force_increment.fill(0.0)

                    for j, section in enumerate(beam_element.cross_sections):
                        section.forces_converged = section.forces
                        section.forces_increment.fill(0.0)
                        section_forces[i, j, :]  = section.forces_converged.reshape(3)
                        section_strains[i, j, :] = section.curvature.reshape(3)

                        section.strains_converged = section.strains
                        section.strains_increment.fill(0.0)

                break
            else:
                convergence_boolean = False
                
        if self.constraint.name == "Displacement control":
            return -u, -llambda, convergence_boolean, section_forces, section_strains
        else:
            return  u,  llambda, convergence_boolean, section_forces, section_strains