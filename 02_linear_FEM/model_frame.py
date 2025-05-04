from print_plot_functions import *
from geometry import *
from mesh import *

import numpy as np

class Frame:
    def __init__(self, column, beam, number_of_stories, story_height, story_width):
        self.column = column
        self.beam   = beam

        self.number_of_stories = number_of_stories
        self.story_height      = story_height
        self.story_width       = story_width

        self.crossSectionProperties = np.array([
            [column.A / 1e6, column.Ix / 1e12],
            [  beam.A / 1e6,   beam.Ix / 1e12]
        ])

        # --- Generate nodes ---
        self.nodes = []
        for i in range(number_of_stories + 1):  # +1 to include the base
            y = i * story_height
            self.nodes.append([0, y])                 # Left column node
            self.nodes.append([story_width, y])       # Right column node
        self.nodes = np.array(self.nodes)

        # --- Generate connectivity ---
        # Convention:
        # Even indices: left nodes (0, 2, 4, ...)
        # Odd indices: right nodes (1, 3, 5, ...)
        self.connectivity = []

        for i in range(number_of_stories):
            bottom_left = 2 * i
            bottom_right = bottom_left + 1
            top_left = bottom_left + 2
            top_right = bottom_right + 2

            # Columns
            self.connectivity.append([bottom_left, top_left])   # Left column
            self.connectivity.append([bottom_right, top_right]) # Right column

            # Beam
            self.connectivity.append([top_left, top_right])

        self.connectivity = np.array(self.connectivity)

        # Assign cross-section indices: 0 for columns, 1 for beams
        self.crossSections = np.zeros((len(self.connectivity), 1), dtype=int)
        for i in range(number_of_stories):
            beam_index = 3 * i + 2
            self.crossSections[beam_index] = 1

        # Degrees of freedom
        self.number_of_dofs = self.nodes.shape[0] * 3

        # Boundary conditions (fully fixed at base)
        self.BCs = np.array([
            [0, 1, 1, 1, 0],   # Node 0 (bottom-left)
            [1, 1, 1, 1, 0]    # Node 1 (bottom-right)
        ])

        plot_input_system(self.nodes, self.connectivity, self.crossSections, self.crossSectionProperties)

        return

    def add_loads(self, nodal_loads, distributed_loads):
        self.nodal_loads = nodal_loads
        self.distributed_loads = distributed_loads
        plot_static_system(self.nodes, self.connectivity, self.crossSections, self.BCs, self.nodal_loads, self.distributed_loads)
        return

    def assemble(self):
        # Initialization
        self.K = np.zeros((self.number_of_dofs, self.number_of_dofs))
        self.F = np.zeros((self.number_of_dofs, 1))

        #Loop over elements to assemble local contributions
        for e, (i, j) in enumerate(self.connectivity):
            
            # 1. Calculate length
            xi, yi = self.nodes[i, :]
            xj, yj = self.nodes[j, :]
            L = np.sqrt((xj-xi)**2+(yj-yi)**2)
            
            # 2. Calculate element local stiffness matrix 
            A     , I    = self.crossSectionProperties[self.crossSections[e]][0]
            qstart, qend = self.distributed_loads[e][0],self.distributed_loads[e][1]
            E = self.column.elements[0].material.E * 1000 * 1000 
            
            ke_local, fe_local = self.get_element_stiffness_and_forces(qstart, qend, E, A, I, L)
            
            cos_theta, sin_theta = (xj-xi)/L, (yj-yi)/L
                
            R = self.LocalRot(sin_theta, cos_theta)
            
            # 4. Rotate element stiffness matrix to global coordinate system
            ke_global = R.T.dot(ke_local).dot(R)
            fe_global = R.T.dot(fe_local)
            
            # 5. Assemble the element global stiffness to the system global stiffness

            self.K[3*i:3*i+3, 3*i:3*i+3] += ke_global[0:3, 0:3]
            self.K[3*i:3*i+3, 3*j:3*j+3] += ke_global[0:3, 3:6]
            self.K[3*j:3*j+3, 3*i:3*i+3] += ke_global[3:6, 0:3]
            self.K[3*j:3*j+3, 3*j:3*j+3] += ke_global[3:6, 3:6]
            self.F[3*i:3*i+3,         0] += fe_global[0:3,   0]
            self.F[3*j:3*j+3,         0] += fe_global[3:6,   0]

        for n, fx, fy, Rtheta in self.nodal_loads:
            self.F[3*int(n)  , 0] += fx
            self.F[3*int(n)+1, 0] += fy
            self.F[3*int(n)+2, 0] += Rtheta

        return

    def solve(self):
        # condensation
        # All degree-of-freedom labels
        allDofs = np.arange(0, self.number_of_dofs)

        restrainedDofs = []
        for i in range (0, self.BCs.shape[0]) :
            for j in range (1, self.BCs.shape[1]-1) :
                if self.BCs[i, j] == 1 :
                    dof = self.BCs[i,0]*3+j-1
                    restrainedDofs.append(dof)
        #Converting list to array
        restrainedDofs = np.array(restrainedDofs)

        freeDofs = np.setdiff1d(allDofs, restrainedDofs)

        Kff = self.K[freeDofs, :][:, freeDofs]
        Kfr = self.K[freeDofs, :][:, restrainedDofs]
        Krf = self.K[restrainedDofs, :][:, freeDofs]
        Krr = self.K[restrainedDofs, :][:, restrainedDofs]

        Ff = self.F[freeDofs, :]

        # solve
        self.U = np.zeros((self.number_of_dofs, 1))

        Uf = np.linalg.solve(Kff, Ff)
        Fr = Krf.dot(Uf)

        self.U[freeDofs] = Uf
        self.F[restrainedDofs] = Fr

        self.U_nodal = np.reshape(self.U, [self.nodes.shape[0], 3])
        return

    def calculate_section_forces(self):
        numberOfElements   = self.connectivity.shape[0]
        self.SectionForces = np.zeros((numberOfElements,6))

        for e, (i, j) in enumerate(self.connectivity):
            # 1. Calculate length    
            xi, yi = self.nodes[i, :]
            xj, yj = self.nodes[j, :]
            L      = np.sqrt((xj-xi)**2+(yj-yi)**2)
            
            # 2. Calculate local element stiffness matrix
            A     , I    = self.crossSectionProperties[self.crossSections[e]][0]
            qstart, qend = self.distributed_loads[e][0],self.distributed_loads[e][1]
            E = self.column.elements[0].material.E * 1000 * 1000 
            
            ke_local, fe_local = self.get_element_stiffness_and_forces(qstart, qend, E, A, I, L)
                
            # 3. Evaluate the rotation matrix
            if L>0:
                cos_theta, sin_theta = (xj-xi)/L, (yj-yi)/L
                
            R = self.LocalRot(sin_theta, cos_theta)
            
            # Choose DOFs needed
            elementDofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]  
            Ue_global = self.U[elementDofs]
            
            # Rotate element global displacements to the local coordinate system
            Ue_local = R.dot(Ue_global)
            
            # Calculate local node forces
            Fe_local = ke_local.dot(Ue_local)
            
            # Save local node forces in matrix
            Ns, Ne = -(Fe_local[0][0]-fe_local[0][0])/1000,  (Fe_local[3][0]-fe_local[3][0])/1000
            Vs, Ve =  (Fe_local[1][0]-fe_local[1][0])/1000, -(Fe_local[4][0]-fe_local[4][0])/1000
            Ms, Me =  (Fe_local[2][0]-fe_local[2][0])/1000, -(Fe_local[5][0]-fe_local[5][0])/1000
            self.SectionForces[e,:]=[Ns,Vs,Ms,Ne,Ve,Me]
        return
    

    def get_element_stiffness_and_forces(self, qstart, qend, E, A, I, L):
        ke_local = self.LocalKBeam(E, A, I, L)
        fe_local = self.LocalFBeam(qstart,qend,L)
        return ke_local, fe_local

    def LocalKBeam(self, E, A, I, L):
        k = np.array([
                [  E*A/L,            0,           0, -E*A/L,            0,           0],
                [      0,  12*E*I/L**3,  6*E*I/L**2,      0, -12*E*I/L**3, 6*E*I/L**2],
                [      0,   6*E*I/L**2,     4*E*I/L,      0,  -6*E*I/L**2,     2*E*I/L],
                [ -E*A/L,            0,           0,  E*A/L,            0,           0],
                [      0, -12*E*I/L**3, -6*E*I/L**2,      0,  12*E*I/L**3, -6*E*I/L**2],
                [      0,   6*E*I/L**2,     2*E*I/L,      0,  -6*E*I/L**2,     4*E*I/L],
        ])
        return k

    def LocalFBeam(self,q_start,q_end,L):
        q=(q_start+q_end)/2
        f = np.array([
                [         0],
                [ q*L   /2 ],
                [ q*L**2/12],
                [         0],
                [ q*L   /2 ],
                [-q*L**2/12],
        ])
        return f
        
    def LocalRot(self,sin_theta, cos_theta):
        R = np.array([
                [ cos_theta, sin_theta,        0,         0,         0,       0],
                [-sin_theta, cos_theta,        0,         0,         0,       0],
                [         0,         0,        1,         0,         0,       0],
                [         0,         0,        0, cos_theta, sin_theta,       0],
                [         0,         0,        0,-sin_theta, cos_theta,       0],
                [         0,         0,        0,         0,         0,       1],
            ]) 
        return R

    def plot_results(self):
        E = self.column.elements[0].material.E * 1000 * 1000 
        plot_results(self.U_nodal, 
                     self.SectionForces, 
                     self.connectivity, 
                     self.nodes, 
                     self.crossSections, 
                     self.distributed_loads, 
                     self.crossSectionProperties, E)
        return