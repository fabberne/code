from .beam_element import *

import numpy as np

class Structure():
    def __init__(self, fixed_DOFs, load_DOFs, nodal_loads):
        self.displacements           = np.zeros((self.number_of_DOFs, 1))
        self.displacements_converged = np.zeros((self.number_of_DOFs, 1))
        self.displacements_increment = np.zeros((self.number_of_DOFs, 1))
        self.lambda_factor           = 0.0
        self.lambda_factor_converged = 0.0
        self.lambda_factor_increment = 0.0


        self.K_global = np.zeros((self.number_of_DOFs, self.number_of_DOFs))
        self.F_global = np.zeros((self.number_of_DOFs, 1))
        self.Residual = np.zeros((self.number_of_DOFs, 1))

        self.fixed_DOFs  = fixed_DOFs
        self.load_DOFs   = load_DOFs
        self.nodal_loads = nodal_loads
        self.apply_boundary_conditions(fixed_DOFs)


    #--------------------------------------------------------------------------------------------------------------------------------#


    def assemble_without_bc(self):
        self.reset_stiffness_and_residual()

        for i, beam_element in enumerate(self.beam_elements):

            K_beam = beam_element.get_global_stiffness_matrix()
            dof_indices = beam_element.beam_DOFs

            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam

        return

    def assemble(self):
        self.reset_stiffness_and_residual()

        for i, beam_element in enumerate(self.beam_elements):
            dof_indices = beam_element.beam_DOFs

            K_beam = beam_element.get_global_stiffness_matrix()
            resisting_forces_beam = beam_element.get_global_resisting_forces()

            self.K_global[np.ix_(dof_indices, dof_indices)] += K_beam
            self.Residual[dof_indices] += resisting_forces_beam

        # Apply boundary conditions
        self.apply_nodal_loads()

        values = np.zeros((len(self.fixed_DOFs), 1))

        self.K_global[self.fixed_DOFs, :] = 0
        self.K_global[:, self.fixed_DOFs] = 0
        self.K_global[self.fixed_DOFs, self.fixed_DOFs] = 1

        self.Residual -= self.F_global * self.lambda_factor
        self.Residual[self.fixed_DOFs] = values
    
    
    #--------------------------------------------------------------------------------------------------------------------------------#


    def apply_boundary_conditions(self, fixed_DOFs):
        # constrain the base
        for dof in fixed_DOFs:
            self.K_global[dof,:] = 0
            self.K_global[:,dof] = 0
            self.K_global[dof,dof] = 1.0
            self.F_global[dof]     = 0.0


    def apply_nodal_loads(self):
        # Apply nodal loads to the global force vector
        self.F_global = np.zeros((self.number_of_DOFs, 1))

        for i, DOF in enumerate(self.load_DOFs):
            self.F_global[DOF] = self.nodal_loads[i]
    
    
    #--------------------------------------------------------------------------------------------------------------------------------#


    def set_displaced_nodes(self, displacements, scale):
        # Set the displaced nodes based on the displacements
        for i, beam_element in enumerate(self.beam_elements):
            beam_element.nodes_displaced = beam_element.nodes_initial + scale * displacements[beam_element.beam_DOFs[[0,1,2,6,7,8]]].reshape(2,3)


    def set_section_max_iter_and_tolerance(self, max_section_iterations, section_tolerance):
        for beam_element in self.beam_elements:
            beam_element.max_section_iterations = max_section_iterations
            for cross_section in beam_element.cross_sections:
                cross_section.tolerance = section_tolerance

                
    def reset_stiffness_and_residual(self):
        self.K_global = np.zeros_like(self.K_global)
        self.F_global = np.zeros_like(self.F_global)
        self.Residual = np.zeros_like(self.Residual)
    
    
    #--------------------------------------------------------------------------------------------------------------------------------#


    def getSystemMatrices(self, displacement_increment, lambda_factor_increment):

        self.displacements_increment = displacement_increment
        self.lambda_factor_increment = lambda_factor_increment

        self.displacements += self.displacements_increment
        self.lambda_factor += self.lambda_factor_increment
        #___ Step 5 - 13 ___
        for i, beam_element in enumerate(self.beam_elements):
            print("         Beam Element ", i+1)
            beam_element.state_determination(-self.displacements_increment[beam_element.beam_DOFs])
        
        #___ Step 14: Assemble the global stiffness matrix and force vector ___
        self.assemble()

        return self.K_global, self.F_global, self.Residual


    def getState(self):
        return self.displacements, self.lambda_factor


#--------------------------------------------------------------------------------------------------------------------------------#


class Frame(Structure):
    def __init__(self, column, beam, number_of_stories, story_height, story_width, number_of_sections_per_elemnt, load_DOFs, nodal_loads):
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
                                                   [self.structure_nodes_initial[bottom_left],
                                                    self.structure_nodes_initial[top_left]],
                                                   left_col_DOFs))
            # add beam
            beam_DOFs      = [top_left  * 6 + 0, top_left  * 6 + 1, top_left  * 6 + 2,
                              top_left  * 6 + 3, top_left  * 6 + 4, top_left  * 6 + 5,
                              top_right * 6 + 0, top_right * 6 + 1, top_right * 6 + 2,
                              top_right * 6 + 3, top_right * 6 + 4, top_right * 6 + 5]
            self.beam_elements.append(Beam_Element(beam,   
                                                   number_of_sections_per_elemnt, 
                                                   [self.structure_nodes_initial[top_left], 
                                                    self.structure_nodes_initial[top_right]],
                                                   beam_DOFs))
            # add right column
            right_col_DOFs = [bottom_right * 6 + 0, bottom_right * 6 + 1, bottom_right * 6 + 2,
                              bottom_right * 6 + 3, bottom_right * 6 + 4, bottom_right * 6 + 5,
                              top_right    * 6 + 0, top_right    * 6 + 1, top_right    * 6 + 2,
                              top_right    * 6 + 3, top_right    * 6 + 4, top_right    * 6 + 5]
            self.beam_elements.append(Beam_Element(column, 
                                                   number_of_sections_per_elemnt, 
                                                   [self.structure_nodes_initial[bottom_right], 
                                                    self.structure_nodes_initial[top_right]],
                                                   right_col_DOFs))

        #--- Apply Boundary Conditions and prepare iteration variables ---                                           
        fixed_DOFs = [0,1,2,3,4,5,6,7,8,9,10,11]
        super().__init__(fixed_DOFs, load_DOFs, nodal_loads)


#--------------------------------------------------------------------------------------------------------------------------------#


class Cantilever(Structure):
    def __init__(self, beam, length, number_of_sections_per_elemnt, load_DOFs, nodal_loads):
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
                                                [self.structure_nodes_initial[0], 
                                                 self.structure_nodes_initial[1]],
                                                [0,1,2,3,4,5,6,7,8,9,10,11]))

        #--- Apply Boundary Conditions and prepare iteration variables ---  
        fixed_DOFs = [0,1,2,3,4,5]
        super().__init__(fixed_DOFs, load_DOFs, nodal_loads)
