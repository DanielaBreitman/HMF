import numpy as np 
import matplotlib.pyplot as plt

import astropy.constants as constants 
from scipy.integrate import simpson

from cosmology import Cosmology



class Transfer():
    """

    Class that calculates the transfer function from Bardeen+86 (T_CDM, eq. 4 in Liddle 96)
    or EH99 (T_master, eq. 24, or T_cbv, eq. 7).

    Assumptions
    -----------
        * Flat cosmology <-> OM_tot = 1
        *

    Parameters
    ----------
    cosmo : object
        Instance of the Cosmology class
    k : np.ndarray
    	Array of wavenumbers k in h / Mpc.
    transfer_fnc_type : str, optional
        'B86'  -> Bardenn+86, using Liddle 96 eq 4 (T_CDM)
        'EH99' -> EH 99 eq 24 (T_master)
        'T_cbv_EH99'  -> EH 99 eq 7 (T_cbv)
        'T_cb_EH99'  -> EH 99 eq 6 (T_cb)
    ### Needed only for EH99 ###
    a_v : float, optional 
        Value of a_v (e.g. a_v = 1 for Fig 1 in EH99)
    B_c : float, optional
    	Value of B_c (e.g. B_c = 1 for Fig 1 in EH99)
    q_eff_is_q : bool, optional
        True if we want q_eff = q (e.g. for Fig 1 in EH99)
        
    Attributes
    ----------
    transfer : np.ndarray
        Transfer function evaluated at array of wavenumbers k 
        for the given cosmology at redshift z.
    
    """
    def __init__(self,
                cosmo : object,
                k : np.ndarray = None, 
                transfer_fnc_type : str = 'B86',
                a_v : float = None, 
                B_c :float = None, 
                q_eff_is_q : bool = False):
        self.cosmo = cosmo
        if k is None:
            self.k = np.logspace(-7, 7, int(1e3))
        else:
            self.k = k

        self.transfer_fnc_type = transfer_fnc_type
        if 'EH99' in transfer_fnc_type:
            self.q_eff_is_q = q_eff_is_q 
            self.a_v = a_v
            self.B_c = B_c
            self.q_eff_is_q = q_eff_is_q
            self.p_cb = self.p(self.cosmo.f_cb)
            self.p_c = self.p(self.cosmo.f_c)
            # As defined in section 3.1 lines 4 - 7 of first paragraph
            # eq 1 from EH 99
            self.z_eq = 2.50e4 * self.cosmo.OMmh2 * self.cosmo.THETA**(-4.)
            # eq 2 from EH 99
            b1 = 0.313 * (self.cosmo.OMmh2) ** (-0.419) * (1. + 0.607 * (self.cosmo.OMmh2) ** 0.674)
            b2 = 0.238 * (self.cosmo.OMmh2) ** 0.223
            z_d = 1291. * (self.cosmo.OMmh2) ** 0.251 / (1. + 0.659 * (self.cosmo.OMmh2) ** 0.828) * (1 + b1 * (self.cosmo.OMbh2) ** b2)
            # eq 3 from EH 99
            self.y_d = (1. + self.z_eq) / (1. + z_d)
            # eq 4 from EH 99
            s = 44.5 * np.log(9.83/ (self.cosmo.OMmh2)) /np.sqrt(1. + 10. * (self.cosmo.OM_b * self.cosmo.h **2) ** (3./4.)) #Mpc 
            # eq 5 from EH 99
            q = self.k * self.cosmo.THETA **2 / (self.cosmo.OMmh2) #/Mpc
            T_m = self.T_master(q, s)
        if 'T_cbv' in transfer_fnc_type:
            T = self.T_cbv(T_m, q)
        elif 'T_cb' in transfer_fnc_type:
            T = self.T_cb(T_m, q)
        elif 'EH99' in transfer_fnc_type:
            T = T_m
        else:
            T = self.T_CDM()

        self.transfer = T


    def p(self, f):
        # Eq. 11 of EH 99
        return max(1./4. * (5. - np.sqrt(1. + 24. * f)), 0)

    def calc_a_v(self):
        # eq. 15 from EH 99 (from HE 98)
        a1 = self.cosmo.f_c / self.cosmo.f_cb * (5. - 2 * (self.p_c + self.p_cb)) / (5. - 4 * self.p_cb)
        a2 = (1. - 0.553 * self.cosmo.f_vb + 0.126 * self.cosmo.f_vb **3.) / (1. - 0.193 * np.sqrt(self.cosmo.f_v * self.cosmo.N_v) +0.169 * self.cosmo.f_v * self.cosmo.N_v**0.2) * (1 + self.y_d) **(self.p_cb - self.p_c)
        a3 = 1 + (self.p_c - self.p_cb) / 2. * (1. + 1./((3. - 4. * self.p_c) * (7. - 4. * self.p_cb))) / (1. + self.y_d)
        a_v = a1 * a2 * a3
        return a_v

        
    def g_squared(self):
        return self.cosmo.OM_m * (1 + self.cosmo.z) **3 + (1 - self.cosmo.OM_m - self.cosmo.OM_L) * (1 + self.cosmo.z)**2 + self.cosmo.OM_L
        
    def D1(self):
        # eq 10 in EH 99
        OMz = self.cosmo.OM_m * (1 + self.cosmo.z)**3 / self.g_squared()
        OMLz = self.cosmo.OM_L / self.g_squared()
        return (1. + self.z_eq) / (1 + self.cosmo.z) * 5./2. * OMz / (OMz**4./7. - OMLz + (1. + OMz/2.) * (1. + OMLz/70.))

    def y_fs(self, q):
        return 17.2 * self.cosmo.f_v * (1. + 0.488 * self.cosmo.f_v ** (-7./6.)) * (self.cosmo.N_v * q / self.cosmo.f_v)**2

    def D_cbv(self, q):
        return (self.cosmo.f_cb ** (0.7 / self.p_cb) + (self.D1()/ (1. + self.y_fs(q)))) **(self.p_cb / 0.7) * self.D1() ** (1. - self.p_cb)


    def D_cb(self, q):
        return ((1. + self.D1())/ (1. + self.y_fs(q))) **(self.p_cb / 0.7) * self.D1() ** (1. - self.p_cb)

    def calculate_T_sup(self, q, s):
        # Calculating T_sup(k)
        if self.a_v is None:
            self.a_v = self.calc_a_v()
        if self.B_c is None:
            self.B_c = 1./(1. - 0.949 * self.cosmo.f_vb)
        if self.q_eff_is_q:
            q_eff = q
        else:
            # Eq 16, 17 from EH 99
            G_eff = self.cosmo.OMmh2 * (np.sqrt(self.a_v) + (1. - np.sqrt(self.a_v)) / (1. + (0.43 * self.k * s)**4.))
            q_eff = (self.k * self.cosmo.THETA**2.) / G_eff #/Mpc
        # eq 19 in EH 99
        L = np.log(np.e + 1.48 * self.B_c * np.sqrt(self.a_v) * q_eff)
        # eq 20 in EH 99
        C = 14.4 + 325. / (1. + 60.5 * q_eff**1.11)
        # eq 18 in EH 99
        T_sup = L / (L + C * q_eff**2)
        return T_sup

    def calculate_B(self, q):
        # Eq 23 in EH 99
        q_v = (3.92 * q * np.sqrt(self.cosmo.N_v)) / self.cosmo.f_v
        # Eq 22 in EH 99
        return 1. + (1.2 * self.cosmo.f_v**0.64 * self.cosmo.N_v**(0.3 + 0.6 * self.cosmo.f_v)) / (q_v**(-1.6) + q_v**0.8)


    def T_master(self, q, s):
        """
        Calculate transfer fnc according to EH 1999 eq. 24
        f_nu = \Omega_\nu / \Omega_0 (ratio of density of massive neutrinos to total Omega 0)
        N_nu = number of most massive neutrino species
        Default values are for Fig 1 of EH 1999
        """
        # Calculate T_sup(k) (eq 18 from EH 99)
        T_sup = self.calculate_T_sup(q, s)
 
        # Calculate B(k) (eq 22 from EH 99)
        B = self.calculate_B(q)

        # Calculate Master transfer fnc (eq 24 from EH 99)
        T_master = T_sup * B

        return T_master

    def T_cbv(self, T_m, q):
        return T_m * self.D_cbv(q) / self.D1() 

    def T_cb(self, T_m, q):
        return T_m * self.D_cb(q) / self.D1() 


    def T_CDM(self):
        G = self.cosmo.OM_m * self.cosmo.h * np.exp(-self.cosmo.OM_b - self.cosmo.OM_b/self.cosmo.OM_m)
        # To have same T_CDM as hmf module, remove factor of h in denominator of q
        q = self.k / ( G)
        t1 = np.log(1. + 2.34 * q) / (2.34 * q) 
        t2 = (1. + 3.89 * q + (16.1 * q) ** 2. + (5.46 * q) ** 3. + (6.71 * q) ** 4.) ** (-1./4.)
        return t1 * t2
