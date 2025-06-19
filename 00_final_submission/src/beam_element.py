from cross_section import *
from scipy.special import legendre

class Beam_Element:
    def __init__(self, geometry, number_of_cross_sections, nodes, beam_DOFs):
        self.number_of_cross_sections = number_of_cross_sections
        self.nodes_initial   = nodes
        self.nodes_displaced = nodes
        
        self.gauss_points, self. gauss_weights = self.get_gauss(number_of_cross_sections)

        self.length = np.linalg.norm(self.nodes_initial[1] - self.nodes_initial[0])

        self.cross_sections = []
        for i in range(number_of_cross_sections):
            self.cross_sections.append(Cross_Section(geometry, self.gauss_points[i], self.gauss_weights[i], self.length))

        # Initialize the local stiffness matrix
        self.K_local = np.zeros((5, 5))

        # Initialize the global stiffness matrix
        self.K_global = np.zeros((12, 12))

        # Initialize the resisting forces and force increment
        self.force_increment            = np.zeros((5, 1))
        self.resisting_forces           = np.zeros((5, 1))
        self.resisting_forces_converged = np.zeros((5, 1))
        self.displacements_residual     = np.zeros((5, 1))

        self.beam_DOFs = np.array(beam_DOFs)


    #--------------------------------------------------------------------------------------------------------------------------------#


    def get_gauss(self, number_of_cross_sections):
        if not isinstance(number_of_cross_sections, int):
            raise ValueError(f"Value must be of type int. given type: {type(number_of_cross_sections)}")
        if number_of_cross_sections < 2 or number_of_cross_sections > 37:
            raise ValueError(f"num can only be between 2 and 37")

        x = list(np.sort(legendre(number_of_cross_sections - 1).deriv().roots))
        x.insert(0, -1)
        x.append(1)
        x = np.array(x)
        w = 2 / (number_of_cross_sections * (number_of_cross_sections - 1) * (legendre(number_of_cross_sections - 1)(x)) ** 2)
        return x, w


    def get_local_stiffness_matrix(self):
        local_flexibility_matrix = np.zeros((5, 5))
        J = self.length / 2

        for cross_section in self.cross_sections:
            local_flexibility_matrix += J * cross_section.gauss_weight * cross_section.get_global_flexibility_matrix()
        
        self.K_local = np.linalg.inv(local_flexibility_matrix)

        return self.K_local


    def get_global_stiffness_matrix(self):
        # Initialize the global stiffness matrix

        # Compute the global stiffness matrix using the local stiffness matrix and transformation matrix
        L = self.get_transformation_matrix()
        Rot = self.get_rotation_matrix()

        self.K_local  = self.get_local_stiffness_matrix()
        self.K_global = L @ self.K_local @ L.T

        # set torsion degrees of freedom to constrained
        self.K_global[3, 3] = 1
        self.K_global[9, 9] = 1

        self.K_global = Rot.T @ self.K_global @ Rot

        return self.K_global

    
    def get_global_resisting_forces(self):
        L   = self.get_transformation_matrix()
        Rot = self.get_rotation_matrix()

        resisting_forces_global = L @ self.resisting_forces
        resisting_forces_global = Rot.T @ resisting_forces_global

        return resisting_forces_global


    def get_transformation_matrix(self):
        # Initialize the transformation matrix
        L = np.zeros((12, 5))

        # --- Forces at node 1 (rows 0:3) ---
        # u1: axial force N
        L[0, 4] = -1.0

        # v1: bending about z-axis → vertical shear from M1z, M2z
        L[1, 2] = 1.0 / self.length
        L[1, 3] = 1.0 / self.length

        # w1: bending about y-axis → lateral shear from M1y, M2y
        L[2, 0] = -1.0 / self.length
        L[2, 1] = -1.0 / self.length

        # No force contribution to θx1 (pure torsion)

        # --- Moments at node 1 (rows 3:6) ---
        # θx1: torsion is not handled here (would come from GJ if implemented separately)
        # θy1: bending about y-axis (vertical bending) comes from M1y
        L[4, 0] = 1.0

        # θz1: bending about z-axis (lateral bending) comes from M1z
        L[5, 2] = 1.0

        # --- Forces at node 2 (rows 6:9) ---
        # u2: axial force N
        L[6, 4] = 1.0

        # v2: bending about z-axis
        L[7, 2] = -1.0 / self.length
        L[7, 3] = -1.0 / self.length

        # w2: bending about y-axis
        L[8, 0] = 1.0 / self.length
        L[8, 1] = 1.0 / self.length

        # --- Moments at node 2 (rows 9:12) ---
        # θy2: bending about y-axis (vertical bending) from M2y
        L[10, 1] = 1.0

        # θz2: bending about z-axis (lateral bending) from M2z
        L[11, 3] = 1.0

        return L

    def get_rotation_matrix(self):

        # Local x-axis (beam axis) = node2 - node1
        v = self.nodes_initial[1] - self.nodes_initial[0]
        L = self.length
        e1 = v / L  # local x-axis (unit vector)

        # Choose a global reference vector for building local z-axis
        g = np.array([0.0, 0.0, 1.0])
        if np.allclose(np.abs(np.dot(e1, g)), 1.0):  # parallel to global Z?
            g = np.array([1.0, 0.0, 0.0])            # switch to global Y

        e2 = np.cross(g, e1)
        e2 /= np.linalg.norm(e2)  # normalize

        e3 = np.cross(e1, e2)     # third orthogonal vector

        # --- Build small 3x3 rotation matrix R ---
        R = np.vstack((e1, e2, e3))  # each row is a local axis

        # --- Build big 12x12 block-diagonal matrix T ---
        T = np.zeros((12, 12))

        for i in range(4):  # 4 blocks (u,v,w θx,θy,θz) at node 1 and node 2
            T[i*3:(i+1)*3, i*3:(i+1)*3] = R

        return T

    
    #--------------------------------------------------------------------------------------------------------------------------------#
    

    def state_determination(self, displacements_increment):
        T = self.get_rotation_matrix()  
        L = self.get_transformation_matrix()      

        displacements_increment_local = T @ displacements_increment
        change_in_displacements_increment = L.T @ displacements_increment_local

        for j in range(self.max_section_iterations):
            print("            Element iteration ", j)
            if j == 0:
                change_in_force_increment = - self.K_local @ change_in_displacements_increment
            else:
                change_in_force_increment = - self.K_local @ self.displacements_residual

            self.force_increment += change_in_force_increment
            self.resisting_forces = self.resisting_forces_converged + self.force_increment
            
            does_element_converge = True
            for cross_section in self.cross_sections:
                if cross_section.state_determination(change_in_force_increment):
                    pass
                else:
                    does_element_converge = False

            self.K_local  = self.get_local_stiffness_matrix()

            if does_element_converge:
                return
            else:
                J = self.length / 2
                self.displacements_residual.fill(0.0)
                for cross_section in self.cross_sections:
                    self.displacements_residual += J * cross_section.gauss_weight * cross_section.get_global_residuals()

        



