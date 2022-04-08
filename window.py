import numpy as np 
import matplotlib.pyplot as plt

import astropy.constants as constants 

from cosmology import Cosmology



class Window(object):
    """

    Class that calculates the window function
    and all related quantities. Supported window 
    function types are: spherical top hat (default), 
    Gaussian or sharp-k.

    Parameters
    ----------
    cosmo : Cosmology object

    k : np.ndarray

    M : float or np.ndarray, optional
        Mass in units of M_sun / h
    R : float or np.ndarray, optional
        Scale in Mpc / h
    window_fnc_type : int, optional
        0 -> spherical top hat (default)
        1 -> Gaussian
        2 -> Sharp-k

    Attributes
    ----------
    mass_to_radius : function
        Convert Mass in M_sun / h to Radius in Mpc / h
    radius_to_mass : function
        Convert Radius in Mpc / h to Mass in M_sun / h
    V_W : function
    	Calculate window function volume in (Mpc / h) ^ 3 
    W_k : function
        Calculate window function
    dW_k_dR : function
    	Calculate derivative of window function wrt R
    	dW_k / dR
    dV_W_dR : function
    	Calculate derivative of volume wrt R
    	dV_W / dR
    dM_dR : function
    	Calculate derivative of Mass wrt R
    	dM / dR


    """
    def __init__(self, 
                cosmo : object,
                k : np.ndarray,
                M : float = None,
                R : float = None,
                window_fnc_type : str = 'spherical'):
        
        # Create Cosmology class object
        self.cosmo = cosmo
        self.k = k


        if window_fnc_type.lower() == 'spherical':
            self.mass_to_radius = self.mass_to_radius_spherical
            self.radius_to_mass = self.radius_to_mass_spherical
            self.V_W = self.V_W_spherical
            self.W_k = self.W_k_spherical
            self.dW_k_dR = self.dW_k_dR_spherical
            self.dV_W_dR = self.dV_W_dR_spherical
            self.dM_dR = self.dM_dR_spherical

        elif window_fnc_type.lower() == 'gaussian':
            self.mass_to_radius = self.mass_to_radius_Gaussian
            self.radius_to_mass = self.radius_to_mass_Gaussian
            self.V_W = self.V_W_Gaussian
            self.W_k = self.W_k_Gaussian
            self.dW_k_dR = self.dW_k_dR_Gaussian
            self.dV_W_dR = self.dV_W_dR_Gaussian
            self.dM_dR = self.dM_dR_Gaussian

        elif window_fnc_type.lower() == 'sharpk':
            self.mass_to_radius = self.mass_to_radius_sharpk
            self.radius_to_mass = self.radius_to_mass_sharpk
            self.V_W = self.V_W_sharpk
            self.W_k = self.W_k_sharpk
            self.dW_k_dR = self.dW_k_dR_sharpk
            self.dV_W_dR = self.dV_W_dR_sharpk
            self.dM_dR = self.dM_dR_sharpk
            
        else:
            raise ValueError('Invalid Window function name. The only options for the window function are Spherical, Gaussian, or sharpk.')
        if R is None and M is not None:
            self.M = M
            self.R = self.mass_to_radius()
        elif R is not None:
            self.R = R
            self.M = self.radius_to_mass()
        else:
            raise ValueError('Provide either the Mass or the Radius.')

    def V_W_spherical(self, R = None):
        if R is None:
            R = self.R
        return  4. * np.pi * R ** 3 / 3.

    def W_k_spherical(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        # Eq 106 in notes
        if type(R) == np.ndarray:
            vol = self.V_W_spherical(R)[:, np.newaxis]
            kR = R[:, np.newaxis] * k
        else:
            vol = self.V_W_spherical(R)
            kR = R * k
        return 3. * ( np.sin(kR) / (kR** 3) - np.cos(kR) / (kR** 2)) 

    def dW_k_dR_spherical(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        if type(R) == np.ndarray:
            kR = R[:, np.newaxis] * k
            k3R4 = (R ** 4)[:, np.newaxis] * k ** 3
        else:
            kR = R * k
            k3R4 = R ** 4 * k ** 3
        return 3. * ((kR ** 2 - 3.) * np.sin(kR) + 3. * kR * np.cos(kR)) / (k3R4)
    
    def dV_W_dR_spherical(self, R = None):
        if R is None:
            R = self.R
        return 4. * np.pi * R ** 2.

    def radius_to_mass_spherical(self, R = None):
        if R is None:
            R = self.R
        return self.V_W_spherical(R) * self.cosmo.mean_density()

    def dM_dR_spherical(self, R = None):
        if R is None:
            R = self.R
        return self.cosmo.mean_density()  * self.dV_W_dR_spherical(R)

    def mass_to_radius_spherical(self, M = None):
        # Mass in M_sun/h
        if M is None:
            M = self.M
        return ( 3. * M / (4. * np.pi * self.cosmo.mean_density()) ) ** (1./3.)

    def V_W_sharpk(self, R = None):
        if R is None:
            R = self.R
        # Eq 107 from notes
        return 6. * np.pi ** 2 * R ** 3

    def W_k_sharpk(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        # Eq 107 from notes
        if type(R) == np.ndarray:
            kR = R[:, np.newaxis] * k
        else:
            kR = R * k
        arr = abs(kR)
        cdn = arr <= 1.
        return cdn.astype(float)

    def dW_k_dR_sharpk(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        if type(R) == np.ndarray:
            kR = R[:, np.newaxis] * k
        else:
            kR = R * k 
        cdn = kR == 1
        return cdn.astype(float)

    def mass_to_radius_sharpk(self, M = None):
        # Mass in M_sun/h
        if M is None:
            M = self.M
        return ( M / (6. * np.pi ** 2 * self.cosmo.mean_density()) ) ** (1./3.)

    def radius_to_mass_sharpk(self, R = None):
        if R is None:
            R = self.R
        return self.V_W_sharpk(R) * self.cosmo.mean_density()

    def dV_W_dR_sharpk(self, R = None):
        if R is None:
            R = self.R
        return 18. * np.pi ** 2 * R **2
    
    def dM_dR_sharpk(self, R = None):
        if R is None:
            R = self.R
        return self.cosmo.mean_density()  * self.dV_W_dR_sharpk(R)

    def V_W_Gaussian(self, R = None):
        if R is None:
            R = self.R
        # Eq 108 in notes
        return (2 * np.pi) ** (3./2.) * R ** 3

    def W_k_Gaussian(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        # Eq 108 in notes
        if type(R) == np.ndarray:
            kR = R[:, np.newaxis] * k
        else:
            kR = R * k
        return np.exp(- kR ** 2 / 2.)

    def radius_to_mass_Gaussian(self, R = None):
        if R is None:
            R = self.R
        return self.V_W_Gaussian(R) * self.cosmo.mean_density()

    def mass_to_radius_Gaussian(self, M = None):
        if M is None:
            M = self.M
        return ( M / ((2. * np.pi) ** (3./2.) * self.cosmo.mean_density()) ) ** (1./3.)
        
    def dW_k_dR_Gaussian(self, k = None, R = None):
        if R is None:
            R = self.R
        if k is None:
            k = self.k
        if type(R) == np.ndarray:
            k2R = R[:, np.newaxis] * k ** 2
        else:
            k2R = R * k ** 2
        return - k2R * self.W_k_Gaussian(k = k, R = R)

    def dV_W_dR_Gaussian(self, R = None):
        if R is None:
            R = self.R
        return 3. * (2. * np.pi) ** (3./2.) * R ** 2.
    
    def dM_dR_Gaussian(self, R = None):
        if R is None:
            R = self.R
        return self.cosmo.mean_density()  * self.dV_W_dR_gaussian(R)

