import numpy as np
from numba import jit

class Material:

    def __init__(self, gamma, E, f_druck, f_zug):

        self.gamma = gamma
        self.E     = E
        self.n     = 0.5

        self.f_druck = f_druck
        self.f_zug   = f_zug

class Concrete_C30_37(Material):

    def __init__(self):
        gamma   = 25 * 10**(-6)  # N/mm3
        E       = 32000          # N/mm2
        f_druck = 20             # N/mm2
        f_zug   = 1.28           # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)
        self.color = (0, 0, 0, 0.5)
        self.name  = "Concrete_C30_37"

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        f_druck = 30     
        f_zug   = 1.28
        e_max   = 0.003
        E = 32000

        a = min(0.7 * f_druck ** (1/15), 1)
        b = max(-0.02 * f_druck ** 0.8, -0.95)
        c = 0.02 * f_druck

        # Vectorized calculation
        stresses = np.where(
            strains <= 0,                       # Negative strain (tensile behavior)
            np.clip(E * strains, -f_zug, 0),  
            
            np.where(
                strains <= e_max,      # Ascending branch
                f_druck * (strains / e_max) ** ((a * (1 - strains / e_max)) /(1 + b * strains / e_max)),

                # Descending branch
                f_druck * (strains / e_max) ** (((c **(e_max / strains)) * (1 - (strains / e_max) ** c)) / (1 + (strains / e_max) ** c))
            )
        )
        
        # Ensure zero stress below tensile failure limit
        stresses[stresses <= -f_zug] = 0

        return stresses

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        # same constants
        f_druck = 30.0
        f_zug   = 1.28
        e_max   = 0.003
        E       = 22800.0

        a = min(0.7 * f_druck ** (1/15), 1.0)
        b = max(-0.02 * f_druck ** 0.8, -0.95)
        c = 0.02 * f_druck

        tangents = np.empty_like(strains)
        for i in range(strains.shape[0]):
            eps = strains[i]
            if eps <= 0.0:
                # tensile: elastic up to failure, then zero
                if eps <= -f_zug / E:
                    tangents[i] = 0.0
                else:
                    tangents[i] = E
            else:
                x = eps / e_max
                if x <= 1.0:
                    # ascending branch: σ = f_druck·xⁿ(x)
                    # with n(x) = a(1–x)/(1+b·x)
                    denom = 1.0 + b * x
                    n     = (a * (1.0 - x)) / denom
                    n_p   = -a * (1.0 + b) / (denom * denom)
                    # dσ/dε = f_druck·xⁿ·[n'(x)·ln x + n/x]·(1/e_max)
                    # guard ln(0) by noting this branch excludes x=0
                    tangents[i] = f_druck * x**n * (n_p * np.log(x) + n / x) / e_max
                else:
                    # descending branch: σ = f_druck·xᵐ(x)
                    # with m(x) = C(x)·D(x)
                    #   C = c^(1/x),  D = (1 - x^c)/(1 + x^c)
                    C   = c ** (1.0 / x)
                    D   = (1.0 - x**c) / (1.0 + x**c)
                    C_p = -np.log(c) / (x*x) * C
                    D_p = -2.0 * c * x**(c - 1.0) / (1.0 + x**c)**2
                    m   = C * D
                    m_p = C_p * D + C * D_p
                    # dσ/dε = f_druck·xᵐ·[m'(x)·ln x + m/x]·(1/e_max)
                    tangents[i] = f_druck * x**m * (m_p * np.log(x) + m / x) / e_max
        return tangents
        
    


class Steel_S235(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 210000          # N/mm2
        f_druck = 235             # N/mm2
        f_zug   = 235             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0, 1, 0.5)
        self.name  = "Steel_S235" 

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E    = 210000
        f_y  = 235
        # yield strain
        e_y  = f_y / E
        # small hardening modulus (e.g. 1% of E)
        H    = 0.01 * E

        abs_eps = np.abs(strains)
        # elastic region mask
        elastic = abs_eps <= e_y

        # allocate
        stress = np.empty_like(strains)
        # elastic
        stress[elastic] = E * strains[elastic]
        # hardening
        idx = ~elastic
        stress[idx] = np.sign(strains[idx]) * (
            f_y + H * (abs_eps[idx] - e_y)
        )
        return stress

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        """
        Tangent dσ/dε:
          = E   for |ε| ≤ ε_y
          = H   for |ε| > ε_y
        """
        E    = 210000
        f_y  = 235
        e_y  = f_y / E
        H    = 0.01 * E

        # piecewise tangent
        return np.where(np.abs(strains) <= e_y, E, H)

class Rebar_B500B(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 205000          # N/mm2
        f_druck = 435             # N/mm2
        f_zug   = 435             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0.2, 1, 0.5)
        self.name  = "Rebar_B500B" 
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_tangent_vectorized(strains):
        """
        Returns dσ/dε for Rebar_B500B:
          E    in elastic region |ε| ≤ ε_s,
          E_h  in hardening region |ε| > ε_s.
        """
        E       = 205000
        f_druck = 500
        # ultimate strain for hardening onset
        e_s     = f_druck / E
        # hardening stiffness
        E_h     = (1.08*f_druck - f_druck) / (0.05 - e_s)

        # piecewise: E or E_h
        return np.where(np.abs(strains) <= e_s, E, E_h)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E       = 205000          # N/mm2
        f_druck = 500             # N/mm2

        f_k  = 1.08 * f_druck
        e_s  = f_druck / E
        E_h  = (f_k - f_druck) / (0.05 - e_s)

        stresses = np.where(
            (np.abs(strains) <= e_s),
            strains * E,
            np.sign(strains) * (f_druck + E_h * (np.abs(strains) - e_s))
        )
        return stresses

class Unknown(Material):

    def __init__(self):

        gamma   = 1
        E       = 1
        f_druck = 1
        f_zug   = 1 

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (1, 0, 0, 0.5)