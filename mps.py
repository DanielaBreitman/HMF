from cosmology import Cosmology
from transfer import Transfer
from window import Window

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as constants 
from scipy.integrate import simpson

class MPS(object):
    """
    Class that calculates \Delta_k^2 (eq 95 in the notes).
    Numerical integration [used to determine the normalisation
    constant A as in eq 109] is done using the Simpson method,
    with resolution determined by the wavenumbers (k) 
    provided as input.

    Assumptions
    -----------
        * Flat cosmology <-> OM_tot = 1
        *
    
    Attributes
    ----------
    cosmo : Instance of Cosmology object, optional
    z : float, optional
        Redshift
    k : np.ndarray, optional
    	Wavenumber in h/Mpc.
    OM_tot : float, optional
        Omega_{total} = Omega_m + Omega_\lambda
        Must be equal to 1 because we assume a flat cosmology.
        Default is 1.
    OM_b : float, optional    
        Density parameter of baryons.
        Default is 0.05.
    OM_v : float, optional
        Density parameter of massive neutrinos.
        Default is 0.2.
    OM_L : float, optional
        Density parameter of cosmological constant
        OM_L = \Lambda / (3H^2)
        Default is 0.
    sig_8 : float, optional
        Normalisation constant of matter power spectrum at z = 0, R = 8h/Mpc 
        assuming spherical top hat window function.
        Default is 0.82.
    h : float, optional
        Little h as in H_0 = 100 h km / s / Mpc
        Default is 0.5.
    N_v : int, optional
        Number of massive neutrino species.
        Default is 1.
    n : float, optional
        Initial power spectrum index
        Default is 1.

    INFO
    ----
    Must provide EITHER Cosmology object OR cosmological parameters 
    to create the cosmology object (at least z).
    """
    def __init__(self, 
    		cosmo : object = None,
                z : float = None,
                k : np.ndarray = None,
                OM_tot : float = 1.,
                OM_b : float = 0.05,
                OM_v : float = 0.2,
                OM_L : float = 0.,
                sigma_8 : float = 0.8159,    
                h : float = 0.5, 
                N_v : int = 1,
                n : float = 0.9667,
                THETA : float = 2.73 / 2.7,
                transfer_fnc_type : str = 'B86',
                a_v : float = None, 
                B_c :float = None, 
                q_eff_is_q : bool = False):
        if k is None:
            self.k = np.logspace(-7, 7, int(1e3))
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
        else:
            self.cosmo = cosmo
        self.transfer_fnc_type = transfer_fnc_type
        self.a_v = a_v
        self.B_c = B_c
        self.q_eff_is_q = q_eff_is_q
        # Create Transfer instance
        self.T = Transfer(self.cosmo, 
                          k = self.k,
                          transfer_fnc_type = transfer_fnc_type,
                          a_v = a_v,
                          B_c = B_c,
                          q_eff_is_q = q_eff_is_q).transfer

        self.sigma_8 = sigma_8
        self.n = n
        # Calculate constant A of matter power spectrum in eq. 109 in course notes
        self.calc_A()
        # Calculate Delta_k^2
        self.mPS = self.Del_squared_k()

    def clone(self, z : float = None):
        """
        Clone and return MPS object
        If z is provided, clone has all same parameters as input (self) 
        but z is updated.
        """
        if z is None:
            cosmo = self.cosmo
        else:
            cosmo = self.cosmo.clone()
            cosmo.update_z(z)
        return MPS(cosmo = cosmo,  k = self.k,
                sigma_8 = self.sigma_8,
                n = self.n,
                transfer_fnc_type = self.transfer_fnc_type,
                a_v = self.a_v,
                B_c = self.B_c,
                q_eff_is_q = self.q_eff_is_q)

    def calc_A(self):
        """
        Calculate normalisation constant A in front of \sigma_k 
        by computing the integral of eq. 104 in course notes
        """
        k = np.logspace(-30,30, int(1e4))
        # Create Transfer instance
        T = Transfer(self.cosmo, 
                          k = k,
                          transfer_fnc_type = self.transfer_fnc_type,
                          a_v = self.a_v,
                          B_c = self.B_c,
                          q_eff_is_q = self.q_eff_is_q).transfer
        window = Window(R = 8, cosmo = self.cosmo,
                        k = self.k, window_fnc_type = 'spherical')
        sig_k_squared = k ** self.n * T ** 2 
        Delta_k_squared_calc_A = k ** 3 * sig_k_squared / (2 * np.pi ** 2)
        integ = window.W_k(k = k, R = 8) ** 2 * Delta_k_squared_calc_A / k
        self.A = np.sqrt(simpson(integ, k))
        return 

    def calc_sig_k_squared(self, k = None, z = None, transfer = None):
        """
        Calculate \sigma_k^2 (z, k ) as in eq. 109 in course notes.

        """
        if z is None:
            z = self.cosmo.z
        if k is None:
            k = self.k
        if transfer is None:
            transfer = self.T
        growth_factor_squared = self.cosmo.D(z) ** 2
        return k ** self.n * transfer ** 2 * growth_factor_squared * (self.sigma_8 / self.A)**2 
        
    def Del_squared_k(self, k = None):
        """
        Calculate normalised matter power spectrum as in eq. 95 in course notes.
        """
        if k is None:
            k = self.k
        # Eq. 95 in notes
        self.sigma_squared_k = self.calc_sig_k_squared(k)
        Delta_k_squared = k ** 3 * self.sigma_squared_k / (2 * np.pi ** 2)
        return Delta_k_squared

