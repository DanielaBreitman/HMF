from cosmology import Cosmology
from transfer import Transfer
from window import Window
from mps import MPS

import matplotlib.pyplot as plt
import numpy as np 

from astropy.constants import G, M_sun
from scipy.integrate import simpson

class HMF(object):
    """
    This class calculates σ^2_M [eq 104 in course notes 
    with numerical integration] 
    and the absolute value of its derivative |dσ_M/dM| 
    [both numerical and analytical differentiation schemes available].
     
    Numerical integration [used to calculate \sigma_M 
    and its derivative] is done using the Simpson method, 
    with resolution determined by the wavenumbers (k) 
    provided as input.
    
    Attributes
    ----------
    
    Mass : float or np.ndarray
        Mass in M_sun / h

    """
    def __init__(self,
                M, # float or array 
                # Cosmology
                cosmo : object = None,
                z : float = None,
                OM_tot : float = 1.,
                OM_b : float = 0.05,
                OM_v : float = 0.2,
                OM_L : float = 0.,
                h : float = 0.5, 
                N_v : int = 1,
                THETA : float = 2.73 / 2.7,
                # Transfer
                k : np.ndarray = None,
                sigma_8 : float = 0.8159,    
                n : float = 0.9667,
                transfer_fnc_type : str = 'B86',
                a_v : float = None, 
                B_c :float = None, 
                q_eff_is_q : bool = False,
                # d \sigma_M/dM calculation method
                method : str = 'analytical',
                window_fnc_type : str = 'spherical'):
        if k is None:
            self.k = np.logspace(-15, 15, int(1e3))
        else:
            self.k = k
        # Create Cosmology instance
        if cosmo is None:
            self.cosmo = Cosmology(z = z,
                          OM_tot = OM_tot,
                          OM_b = OM_b,
                          OM_v = OM_v,
                          OM_L = OM_L,
                          h = h, 
                          N_v = N_v,
                          THETA = THETA)
            cosmo0 = Cosmology(z = 0,
                          OM_tot = OM_tot,
                          OM_b = OM_b,
                          OM_v = OM_v,
                          OM_L = OM_L,
                          h = h, 
                          N_v = N_v,
                          THETA = THETA)
        else:
            self.cosmo = cosmo
            cosmo0 = cosmo.clone()
            cosmo0.update_z(z = 0)
        # Create MPS instance
        # We calculate MPS at z = 0 because we extrapolate from z = 0
        # The extrapolation to higher z is taken into account in the \delta_c = 1.686/D(z)
        self.mps_obj = MPS(cosmo = cosmo0, k = self.k, n = n, sigma_8 = sigma_8, 
                      transfer_fnc_type = transfer_fnc_type,
                      a_v = a_v,
                      B_c = B_c,
                      q_eff_is_q = q_eff_is_q)
        self.Delta_k_squared = self.mps_obj.mPS
        
        # Take transfer instance from MPS instance
        self.transfer = self.mps_obj.T
        self.M = np.array([M]).flatten()
        # Create Window instance
        self.window = Window(M = self.M, cosmo = self.cosmo,
                        k = self.k, window_fnc_type = window_fnc_type)
        self.method = method
        self.n = n
        self.R = self.window.mass_to_radius(M) 
        self.dn_dM = self.calc_hmf()

    def sig_squared_M(self, R = None, z = None):
        """
        Returns
        -------
        \sigma_M^2 (eq. 104 from course notes)
        """
        if R is None:
            R = self.R
        if z is not None:
            Delta_k_squared = self.mps_obj.clone(z = z).mPS
        else:
            Delta_k_squared = self.Delta_k_squared
        # Compute the integral
        integrand = self.window.W_k(self.k, R) ** 2 * Delta_k_squared / self.k
        integral = simpson(integrand, self.k)
        return integral

    def derivative(self, z = None):
        """
        Returns
        -------
        d \sigma_M / dM [numerical or analytic]
        """
        if z is not None:
            sig2_M = self.sig_squared_M(z = z)
            Delta_k_squared = self.mps_obj.clone(z = z).mPS
        else:
            sig2_M = self.sig_2
            Delta_k_squared = self.Delta_k_squared

        if self.method == 'numerical':
            dsig2_M_dR = self.deriv(self.sig_squared_M, self.R)
            return abs(dsig2_M_dR)/(2.* sig2_M**0.5 * self.window.dM_dR())
        elif self.method == 'analytical':
            integrand = Delta_k_squared * self.window.W_k(self.k, self.R) * self.window.dW_k_dR(self.k, self.R) / self.k
            dsig2_M_dR = simpson(integrand, self.k)
            return abs(dsig2_M_dR / (self.window.dM_dR() * sig2_M ** 0.5))
        else:
            raise ValueError('No such method implemented. Choose between analytical and numerical.')

    def calc_hmf(self):
        """
    	Returns
    	-------
    	dn / dM (eq 116 in course notes)
    	"""
        d_c = self.cosmo.delta_c()
        self.sig_2 = self.sig_squared_M()
        self.dsig_M_dM = self.derivative()
        self.hmf = np.sqrt(2. / np.pi) * d_c / self.sig_2 * self.dsig_M_dM * np.exp(- d_c ** 2 / (2. * self.sig_2))
        dn_dM = self.hmf * self.cosmo.mean_density() / self.M
        return dn_dM

    def deriv(self, f, a, method = 'central', h = 1e-5):
        """ Difference formula for f'(a) with step size h.

        Parameters
        ----------
        f : function
            Vectorized function of one variable
        a : np.ndarray
            Where to compute derivative
        method : string, optional
            Difference formula: 'forward', 'backward' or 'central'
            Default is 'central'.
        h : float, optional
            Step size in difference formula.
            Default is 1e-5.

        Returns
        -------
        np.ndarray
        """
        if method == 'central':
            return (f(a + h) - f(a - h))/(2*h)
        elif method == 'forward':
            return (f(a + h) - f(a))/h
        elif method == 'backward':
            return (f(a) - f(a - h))/h
        else:
            raise ValueError("Method must be 'central', 'forward' or 'backward'.")
