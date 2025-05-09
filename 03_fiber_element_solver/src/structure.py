from beam_element import *

import numpy as np

class Frame():
    def __init__(self, column, beam, number_of_stories, story_height, story_width, number_of_sections_per_elemnt):
        self.number_of_stories = number_of_stories
        self.story_height      = story_height
        self.story_width       = story_width

        self.number_of_nodes = 2 * (self.number_of_stories + 1)
        self.number_of_DOFs  = 6 * self.number_of_nodes

        # --- Generate nodes ---
        self.structure_nodes_initial = []
        for i in range(number_of_stories + 1):
            z = i * story_height
            self.structure_nodes_initial.append([0, 0, z])                 
            self.structure_nodes_initial.append([story_width, 0, z])
        self.structure_nodes_initial = np.array(self.structure_nodes_initial)

        #--- Generate Beam Elements ---
        self.beam_elements = []
        for i in range(number_of_stories):
            bottom_left  = 2 * i
            bottom_right = bottom_left + 1
            top_left     = bottom_left + 2
            top_right    = bottom_right + 2

            # add left column
            left_col_DOFs  = [bottom_left * 6 + 0, bottom_left * 6 + 1, bottom_left * 6 + 2,
                              bottom_left * 6 + 3, bottom_left * 6 + 4, bottom_left * 6 + 5,
                              top_left    * 6 + 0, top_left    * 6 + 1, top_left    * 6 + 2,
                              top_left    * 6 + 3, top_left    * 6 + 4, top_left    * 6 + 5]
            self.beam_elements.append(Beam_Element(column, 
                                                   number_of_sections_per_elemnt, 
                                                   [self.structure_nodes_initial[bottom_left], self.structure_nodes_initial[top_left]],
                                                   left_col_DOFs))
            # add beam
            beam_DOFs      = [top_left  * 6 + 0, top_left  * 6 + 1, top_left  * 6 + 2,
                              top_left  * 6 + 3, top_left  * 6 + 4, top_left  * 6 + 5,
                              top_right * 6 + 0, top_right * 6 + 1, top_right * 6 + 2,
                              top_right * 6 + 3, top_right * 6 + 4, top_right * 6 + 5]
            self.beam_elements.append(Beam_Element(beam,   
                                                   number_of_sections_per_elemnt, 
                                                   [self.structure_nodes_initial[top_left], self.structure_nodes_initial[top_right]],
                                                   beam_DOFs))
            # add right column
            right_col_DOFs = [bottom_right * 6 + 0, bottom_right * 6 + 1, bottom_right * 6 + 2,
                              bottom_right * 6 + 3, bottom_right * 6 + 4, bottom_right * 6 + 5,
                              top_right    * 6 + 0, top_right    * 6 + 1, top_right    * 6 + 2,
                              top_right    * 6 + 3, top_right    * 6 + 4, top_right    * 6 + 5]
            self.beam_elements.append(Beam_Element(column, 
                                                   number_of_sections_per_elemnt, 
                                                   [self.structure_nodes_initial[bottom_right], self.structure_nodes_initial[top_right]],
                                                   right_col_DOFs))

        #--- set variable for NR solver ---
        self.displacements           = np.zeros((self.number_of_DOFs, 1))
        self.displacements_converged = np.zeros((self.number_of_DOFs, 1))
        self.displacements_increment = np.zeros((self.number_of_DOFs, 1))
        self.lambda_factor           = 0.0
        self.lambda_factor_converged = 0.0
        self.lambda_factor_increment = 0.0


        self.K_global = np.zeros((self.number_of_DOFs, self.number_of_DOFs))
        self.F_global = np.zeros((self.number_of_DOFs, 1))
        self.Residual = np.zeros((self.number_of_DOFs, 1))
        self.resisting_forces = np.zeros((self.number_of_DOFs, 1))

        #--- Apply Boundary Conditions ---
        self.apply_boundary_conditions()

    def assemble_stiffness_and_forces(self):
        # Assemble global stiffness matrix and force vector
        for i, beam_element in enumerate(self.beam_elements):
            # Get local stiffness matrix and force vector
            K_beam = beam_element.get_global_stiffness_matrix()
            F_beam = beam_element.get_global_force_vector()

            # Get global DOF indices for the element
            dof_indices = beam_element.beam_DOFs

            # Assemble into global stiffness matrix and force vector
            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam
            self.F_global[dof_indices] += F_beam

        return

    def assemble(self):
        self.reset_stiffness_and_residual()

        for i, beam_element in enumerate(self.beam_elements):
            # Get global DOF indices for the element
            dof_indices = beam_element.beam_DOFs

            # Get local stiffness matrix and force vector
            K_beam = beam_element.get_global_stiffness_matrix()
            resisting_forces_beam = beam_element.get_global_resisting_forces()

            # Assemble into global stiffness matrix and force vector
            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam
            self.resisting_forces[dof_indices] += resisting_forces_beam

        # Apply boundary conditions
        self.apply_nodal_loads([2*6], [1])

        fixed_dofs = [0,1,2,3,4,5,6,7,8,9,10,11]
        values = np.zeros((len(fixed_dofs), 1))

        self.K_global[fixed_dofs, :] = 0
        self.K_global[:, fixed_dofs] = 0
        self.K_global[fixed_dofs, fixed_dofs] = 1

        self.Residual = self.resisting_forces - self.F_global * self.lambda_factor
        self.Residual[fixed_dofs] = values

    def reset_stiffness_and_residual(self):
        self.K_global = np.zeros_like(self.K_global)
        self.F_global = np.zeros_like(self.F_global)
        self.Residual = np.zeros_like(self.Residual)
        self.resisting_forces = np.zeros_like(self.resisting_forces)

    def apply_boundary_conditions(self):
        # constrain the base
        fixed_dofs = [0,1,2,3,4,5,6,7,8,9,10,11]
        for dof in fixed_dofs:
            self.K_global[dof,:] = 0
            self.K_global[:,dof] = 0
            self.K_global[dof,dof] = 1.0
            self.F_global[dof]     = 0.0

    def apply_nodal_loads(self, DOFs, loads):
        # Apply nodal loads to the global force vector
        self.F_global = np.zeros((self.number_of_DOFs, 1))
        for i, dof in enumerate(DOFs):
            self.F_global[dof] = loads[i]

    def set_displaced_nodes(self, displacements, scale):
        # Set the displaced nodes based on the displacements
        for i, beam_element in enumerate(self.beam_elements):
            beam_element.nodes_displaced = beam_element.nodes_initial + scale*displacements[beam_element.beam_DOFs[[0,1,2,6,7,8]]].reshape(2,3)

    def getSystemMatrices(self, displacement_increment, lambda_factor_increment):

        self.displacements_increment = displacement_increment
        self.lambda_factor_increment = lambda_factor_increment

        self.displacements += self.displacements_increment
        self.lambda_factor += self.lambda_factor_increment
        #___ Step 5 - 13 ___
        for i, beam_element in enumerate(self.beam_elements):
            print("         Beam Element ", i+1)
            beam_element.state_determination(-self.displacements_increment[beam_element.beam_DOFs])
            #print("         Beam Element ", i, " converged")
        
        #___ Step 14: Assemble the global stiffness matrix and force vector ___
        self.assemble()

        return self.K_global, self.F_global, self.Residual

    def getState(self):
        return self.displacements, self.lambda_factor


class Cantilever():
    def __init__(self, beam, length, number_of_sections_per_elemnt):
        self.number_of_nodes = 2
        self.number_of_DOFs  = 6 * self.number_of_nodes

        # --- Generate nodes ---
        self.structure_nodes_initial = []
        self.structure_nodes_initial.append([0, 0, 0])                 
        self.structure_nodes_initial.append([length, 0, 0])
        self.structure_nodes_initial = np.array(self.structure_nodes_initial)

        #--- Generate Beam Elements ---
        self.beam_elements = []
        self.beam_elements.append(Beam_Element(beam, 
                                                number_of_sections_per_elemnt, 
                                                [self.structure_nodes_initial[0], self.structure_nodes_initial[1]],
                                                [0,1,2,3,4,5,6,7,8,9,10,11]))

        #--- set variable for NR solver ---
        self.displacements           = np.zeros((self.number_of_DOFs, 1))
        self.displacements_converged = np.zeros((self.number_of_DOFs, 1))
        self.displacements_increment = np.zeros((self.number_of_DOFs, 1))
        self.lambda_factor           = 0.0
        self.lambda_factor_converged = 0.0
        self.lambda_factor_increment = 0.0


        self.K_global = np.zeros((self.number_of_DOFs, self.number_of_DOFs))
        self.F_global = np.zeros((self.number_of_DOFs, 1))
        self.Residual = np.zeros((self.number_of_DOFs, 1))
        self.resisting_forces = np.zeros((self.number_of_DOFs, 1))

        #--- Apply Boundary Conditions ---
        self.apply_boundary_conditions()

    def assemble_stiffness_and_forces(self):
        # Assemble global stiffness matrix and force vector
        for i, beam_element in enumerate(self.beam_elements):
            # Get local stiffness matrix and force vector
            K_beam = beam_element.get_global_stiffness_matrix()
            F_beam = beam_element.get_global_force_vector()

            # Get global DOF indices for the element
            dof_indices = beam_element.beam_DOFs

            # Assemble into global stiffness matrix and force vector
            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam
            self.F_global[dof_indices] += F_beam

        return

    def assemble(self):
        self.reset_stiffness_and_residual()

        for i, beam_element in enumerate(self.beam_elements):
            # Get global DOF indices for the element
            dof_indices = beam_element.beam_DOFs

            # Get local stiffness matrix and force vector
            K_beam = beam_element.get_global_stiffness_matrix()
            resisting_forces_beam = beam_element.get_global_resisting_forces()

            # Assemble into global stiffness matrix and force vector
            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam
            self.resisting_forces[dof_indices] += resisting_forces_beam

        # Apply essential conditions
        self.apply_nodal_loads([6+2], [1])

        fixed_dofs = [0,1,2,3,4,5]
        values = np.zeros((len(fixed_dofs), 1))

        self.K_global[fixed_dofs, :] = 0
        self.K_global[:, fixed_dofs] = 0
        self.K_global[fixed_dofs, fixed_dofs] = 1

        self.Residual = self.resisting_forces - self.F_global * self.lambda_factor
        self.Residual[fixed_dofs] = values

    def reset_stiffness_and_residual(self):
        self.K_global = np.zeros_like(self.K_global)
        self.F_global = np.zeros_like(self.F_global)
        self.Residual = np.zeros_like(self.Residual)
        self.resisting_forces = np.zeros_like(self.resisting_forces)

    def apply_boundary_conditions(self):
        # constrain the base
        fixed_dofs = [0,1,2,3,4,5]
        for dof in fixed_dofs:
            self.K_global[dof,:] = 0
            self.K_global[:,dof] = 0
            self.K_global[dof,dof] = 1.0
            self.F_global[dof]     = 0.0

    def apply_nodal_loads(self, DOFs, loads):
        # Apply nodal loads to the global force vector
        self.F_global = np.zeros((self.number_of_DOFs, 1))
        for i, dof in enumerate(DOFs):
            self.F_global[dof] = loads[i]

    def set_displaced_nodes(self, displacements, scale):
        # Set the displaced nodes based on the displacements
        for i, beam_element in enumerate(self.beam_elements):
            beam_element.nodes_displaced = beam_element.nodes_initial + scale*displacements[beam_element.beam_DOFs[[0,1,2,6,7,8]]].reshape(2,3)

    def getSystemMatrices(self, displacement_increment, lambda_factor_increment):

        self.displacements_increment = displacement_increment
        self.lambda_factor_increment = lambda_factor_increment

        self.displacements += self.displacements_increment
        self.lambda_factor += self.lambda_factor_increment
        #___ Step 5 - 13 ___
        for i, beam_element in enumerate(self.beam_elements):
            print("         Beam Element ", i+1)
            beam_element.state_determination(-self.displacements_increment[beam_element.beam_DOFs])
            #print("         Beam Element ", i, " converged")
        
        #___ Step 14: Assemble the global stiffness matrix and force vector ___
        self.assemble()

        return self.K_global, self.F_global, self.Residual

    def getState(self):
        return self.displacements, self.lambda_factor

