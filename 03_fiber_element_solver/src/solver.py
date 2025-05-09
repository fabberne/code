import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import tqdm
import abc
import sys
import structure
from print_color import print

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
        
        dof_labels = [
            'u [mm]', 'v [mm]', 'w [mm]', 'θx [rad]', 'θy [rad]', 'θz [rad]'
        ]
        F_int_labels = [
            'Fx [kN]', 'Fy [kN]', 'Fz [kN]', 'Mx [kNm]', 'My [kNm]', 'Mz [kNm]'
        ]

        for node in range(self.structure.number_of_nodes):
            u_node     = self.displacements[node*6:(node+1)*6].reshape(6)
            F_int_node = self.forces[node*6:(node+1)*6].reshape(6)

            F_int_node[[0,1,2]] = F_int_node[[0,1,2]] / 1000.0 # Convert to kN
            F_int_node[[3,4,5]] = F_int_node[[3,4,5]] / 1000000.0 # Convert to kNm

            # Create a DataFrame
            df1 = pd.DataFrame({
                'Displacement': u_node
            }, index=dof_labels)
            df2 = pd.DataFrame({
                'Internal Force': F_int_node
            }, index=F_int_labels)

            # Format nicely
            pd.set_option('display.precision', 9)
            pd.set_option('display.float_format', '{:,.6f}'.format)
            print("--------------------------------------")
            print("Node", node)
            print(df1)
            print(df2)


class Linear(Solver):
    def __init__(self, structure):
        self.structure = structure

        self.displacements = np.zeros(self.structure.number_of_DOFs)
        self.forces        = np.zeros(self.structure.number_of_DOFs)

    def solve(self, load_DOFs, nodal_loads):
        self.structure.assemble_stiffness_and_forces()
        self.structure.apply_boundary_conditions()

        self.structure.apply_nodal_loads(load_DOFs, nodal_loads)

        self.displacements = np.linalg.solve(self.structure.K_global, self.structure.F_global)

        self.structure.assemble_stiffness_and_forces()
        self.forces = np.dot(self.structure.K_global, self.displacements)

        self.print_nodal_displacements_and_forces()

        return
    



class Nonlinear(Solver):
    def __init__(self, structure, constraint="Load", NR_tolerance=1e-6, NR_max_iter=1000, section_tolerance=1e-6, section_max_iter=1000):
        self.structure = structure

        self.attempts = 5

        self.NR_tolerance = NR_tolerance
        self.NR_max_iter  = NR_max_iter

        self.section_tolerance = section_tolerance
        self.section_max_iter  = section_max_iter

        self.iteration = 0
        self.iteration_section = 0

        self.setConstraint(constraint)

    def setConstraint(self, constraint):

        """
        Specify the type of constraint.

        Parameters
        ----------
        constraint: str
                The type of constraint to be used.

        Raises
        ------
        ValueError
                If an invalid type of constraint is specified.
        """

        self.constraint = Constraint(constraint)

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
            lambda0 = float(lambda0)
            u0 = u0.copy()


            while (not convergence_boolean) and (attempt <= self.attempts):
                print("   Attempt ", attempt)
                u, llambda, convergence_boolean = self.getSolution(
                    u0, lambda0, increments[step]
                )

                attempt += 1

                if (not convergence_boolean) and (attempt <= self.attempts):
                    increments[step] *= 0.5
                    print("   Decreased increment to ", increments[step])

            if not convergence_boolean:
                print("   Failed to reach convergence after ", attempt-1, "attempts")

            u_history[step + 1, :]   = u.reshape(self.structure.number_of_DOFs)
            lambda_history[step + 1] = llambda

        return u_history, lambda_history


    def getSolution(self, u0, lambda0, increment):

        #self.structure.lambda_factor = lambda0
        self.structure.assemble()
        Stiffness_K, fext, ResidualsR = self.structure.K_global, self.structure.F_global, self.structure.Residual
        convergence_norm = max(np.linalg.norm(fext),1)

        #___ # Step 1: Predict the solution at the beginning of the step ___
        (
            u,
            llambda,
            deltaUp,
            deltalambdap,
            Stiffness_K,
            fext,
            ResidualsR,
        ) = self.constraint.predict(
            self.structure.getSystemMatrices,
            u0,
            lambda0,
            increment,
            Stiffness_K,
            fext,
            ResidualsR,
        )
        
        for iteration in range(self.NR_max_iter):

            #___ Step 2: Calculate the deformation increments ___
            print("      NR Iteration ", iteration)
            g, h, s = self.constraint.get(
                u, llambda, u0, lambda0, deltaUp, deltalambdap, increment
            )

            du_tilde        =   np.linalg.inv(Stiffness_K).dot(fext)
            du_double_tilde = - np.linalg.inv(Stiffness_K).dot(ResidualsR)

            deltalambdap = - (g + np.transpose(h).dot(du_double_tilde)) / (s +np.transpose(h).dot(du_tilde))
            deltaUp      = deltalambdap * du_tilde + du_double_tilde
            
            # Update the solution variables
            #u, llambda = u + deltaUp, llambda + deltalambdap

            #___ Steps 3 - 14 ___
            Stiffness_K, fext, ResidualsR = self.structure.getSystemMatrices(deltaUp, deltalambdap)
        
            u       = self.structure.displacements
            llambda = self.structure.lambda_factor
            

            #___ Step 15: Check convergence criteria ___
            print("      Residuals Norm ", np.linalg.norm(ResidualsR), color="red")

            if np.linalg.norm(ResidualsR) <= self.NR_tolerance * convergence_norm:
                print("NR Converged!")
                convergence_boolean = True
                #u       = self.structure.displacements
                #llambda = self.structure.lambda_factor

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
        return u, llambda, convergence_boolean


class Constraint:

    """
    Description

    Parameters
    ----------
    constraint: str
            The type of constraint to be used
                    'Arc'          : Arc-length method
                    'Displacement' : Displacement control
                    'Load'         : Load control
                    'Riks'         : Riks method

    Attributes
    ----------
    name: str
            The name of the constraint function.

    Methods
    -------
    get(x, c, x0, c0, dx, dc, S, length, *args)
            Get the value of the constraint function and its derivatives.
    predict(func, x, c, A, b, r)
            Predict the solution at the beginning of the step.

    Raises
    ------
    KeyError
            If an invalid type of constraint is specified.
    """

    __constraints = {
        "Displacement": "_Displacement",
        "Load": "_Load",
        "Arc": "_Arc",
        "Riks": "_Riks",
    }

    def __init__(self, constraint):

        module = sys.modules[__name__]

        try:
            constraint = getattr(module, self.__constraints[constraint])
        except KeyError:
            raise KeyError("Invalid type of constraint")

        self._constraint = constraint()

    def get(
        self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None, *args
    ):
        """
        Get the values of the constraint function and the gradients with
        respect to the independent variable and the scale factor.

        Parameters
        ----------
        u: ndarray
                The current value of the independent variable.
        llambdas: ndarray
                The current value of the scale factor.
        u0: ndarray
                The value of the independent variable at the beginning of the step.
        lambda0: ndarray
                The value of the scale factor at the beginning of the step.
        deltaUp: ndarray
                The increment of the independent variable.
        deltalambdap: ndarray
                The increment of the scale factor.
        T: ndarray
                The selection matrix of the independent variable.
        deltaS: float
                The increment of the arc length.

        Returns
        -------
        g: float
                The value of the constraint function.
        h: ndarray
                The derivative of the constraint function with respect to the
                independent variable.
        s: float
                The derivative of the constraint function with respect to the
                scaling factor.
        """

        g, h, s = self._constraint.get(
            u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None, *args
        )

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        """
        Get the prediction at the beginning of each solution step.

        Parameters
        ----------
        func: callable
                ...
        u: ndarray
                The initial value of the solution variable.
        llambda: float
                The initial value of the scale factor.
        deltaS: float
                The increment of the arc length.
        StiffnessK: ndarray
                The tangent coefficient matrix.
        fext: ndarray
                The constant vector.
        Residualsr: ndarray
                The residual vector.

        Returns
        -------
        u: ndarray
                The solution variable at the prediction point.
        llambda: float
                The scale factor at the prediction point.
        deltaUp: ndarray
                The predicted solution increment.
        deltalambdap: float
                The predicted scale factor increment.
        StiffnessK: ndarray
                The tangent coefficient matrix at the prediciton point.
        fext: ndarray
                The constant vector at the prediction point.
        Residualsr: ndarray
                The residual vector at the prediction point.
        """

        (
            u,
            llambda,
            deltaUp,
            deltalambdap,
            StiffnessK,
            fext,
            Residualsr,
        ) = self._constraint.predict(
            func, u, llambda, deltaS, StiffnessK, fext, Residualsr
        )

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr

    @property
    def name(self):
        return self._constraint.name


class _Constraint(abc.ABC):
    @abc.abstractmethod
    def get(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class _Load(_Constraint):

    name = "Load control"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        g = llambda - llambda0 - deltaS
        h = np.zeros_like(u)
        s = 1

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp, deltalambdap = np.zeros_like(u), 0

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Displacement(_Constraint):

    name = "Displacement control"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):
        if T is None:
            T     = np.zeros_like(u).reshape(len(u))
            T[8] = 1

        g = T.dot(u - (u0 - deltaS))
        h = T
        s = 0

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp, deltalambdap = np.zeros_like(u), 0

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Riks(_Constraint):

    name = "Riks"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        g = deltaUp.T.dot(u - u0 - deltalambdap * deltaUp) + deltalambdap * (llambda - llambda0 - deltalambdap)
        h = deltaUp
        s = deltalambdap

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp = np.linalg.inv(StiffnessK).dot(fext)
        
        kappa = np.transpose(fext).dot(deltaUp) / np.transpose(deltaUp).dot(deltaUp)
        
        deltalambdap = np.sign(kappa) * deltaS / np.linalg.norm(deltaUp)

        u, llambda = u + deltalambdap * deltaUp, llambda + deltalambdap
        StiffnessK, fext, Residualsr = func(deltaUp, deltalambdap)

        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr


class _Arc(_Constraint):

    name = "Arc-length"

    def get(self, u, llambda, u0, llambda0, deltaUp, deltalambdap, deltaS, T=None):

        g = np.sqrt(np.transpose(u - u0).dot(u - u0) + (llambda - llambda0)**2) - deltaS
        h = (u - u0) / g
        s = (llambda - llambda0) / g

        return g, h, s

    def predict(self, func, u, llambda, deltaS, StiffnessK, fext, Residualsr):

        deltaUp = np.linalg.inv(StiffnessK).dot(fext)
        
        kappa = np.transpose(fext).dot(deltaUp) / np.transpose(deltaUp).dot(deltaUp)
        
        deltalambdap = np.sign(kappa) * deltaS / np.linalg.norm(deltaUp)

        u, llambda = u + deltalambdap * deltaUp, llambda + deltalambdap
        StiffnessK, fext, Residualsr = func(deltaUp, deltalambdap)
        
        return u, llambda, deltaUp, deltalambdap, StiffnessK, fext, Residualsr