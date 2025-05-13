from abc import ABC, abstractmethod
import numpy as np


class Constraint(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def predict(self):
        pass



class Load(Constraint):
    name = "Load control"


    def get(self, u, llambda, u_0, llambda_0, delta_u_p, delta_lambda_p, delta_s, T=None, controlled_DOF=None):
        g = llambda - llambda_0 - delta_s
        h = np.zeros_like(u)
        s = 1

        return g, h, s


    def predict(self, func, u, llambda, delta_s, stiffness_K, f_ext, residuals_R):
        delta_u_p, delta_lambda_p = np.zeros_like(u), 0

        return u, llambda, delta_u_p, delta_lambda_p, stiffness_K, f_ext, residuals_R



class Displacement(Constraint):
    name = "Displacement control"


    def get(self, u, llambda, u_0, llambda_0, delta_u_p, delta_lambda_p, delta_s, T=None, controlled_DOF=None):
        if T is None:
            T = np.zeros_like(u).reshape(len(u))
            T[controlled_DOF] = 1

        g = T.dot(u - (u_0 - delta_s))
        h = T
        s = 0

        return g, h, s


    def predict(self, func, u, llambda, delta_s, stiffness_K, f_ext, residuals_R):
        delta_u_p, delta_lambda_p = np.zeros_like(u), 0

        return u, llambda, delta_u_p, delta_lambda_p, stiffness_K, f_ext, residuals_R



class Arc(Constraint):
    name = "Arc-length"


    def get(self, u, llambda, u_0, llambda_0, delta_u_p, delta_lambda_p, delta_s, T=None, controlled_DOF=None):
        g = np.sqrt(np.transpose(u - u_0).dot(u - u_0) + (llambda - llambda_0)**2) - delta_s
        h = (u - u_0) / g
        s = (llambda - llambda_0) / g

        return g, h, s


    def predict(self, func, u, llambda, delta_s, stiffness_K, f_ext, residuals_R):
        delta_u_p = np.linalg.solve(stiffness_K, f_ext)
        
        kappa = np.transpose(f_ext).dot(delta_u_p) / np.transpose(delta_u_p).dot(delta_u_p)
        
        delta_lambda_p = np.sign(kappa) * delta_s / np.linalg.norm(delta_u_p)

        u, llambda = u + delta_lambda_p * delta_u_p, llambda + delta_lambda_p
        stiffness_K, f_ext, residuals_R = func(delta_u_p, delta_lambda_p)
        
        return u, llambda, delta_u_p, delta_lambda_p, stiffness_K, f_ext, residuals_R