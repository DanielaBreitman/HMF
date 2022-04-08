import numpy as np 
import matplotlib.pyplot as plt

class Cosmology:

    """
    Class that contains all information pertaining to the cosmology
    Assumptions
    -----------
        * Flat cosmology <-> OM_tot = 1
        *

    Attributes
    ----------
    z : float
        Redshift
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
    debug : bool, optional
        If True, displays step-by-step plots.
        Default is False.
    """
    def __init__(self, 
                z : float,
                OM_tot : float = 1.,
                OM_b : float = 0.05,
                OM_v : float = 0.2,
                OM_L : float = 0.,
                h : float = 0.5, 
                N_v : int = 1,
                THETA : float = 2.73 / 2.7,
                debug : bool = False):

        ########################
        #OM_m = OM_cdm + OM_b + OM_v
        #OM_tot = OM_m + OM_L
        #EH99 Mixed dark matter (MDM) => OM_m = 1, OM_b = 0.05, OM_v = 0.2, 
        #                                OM_cdm = 1 - (OM_b + OM_cdm), h = 0.5,
        #                                N_v = 1   
        ######################

        self.z = z
        #OM_cdm = OM_tot - (OM_L + OM_b + OM_v)        
        # Baryons
        self.OM_b = OM_b
        # Massive neutrinos
        self.OM_v = OM_v
        # Cosmological cst
        self.OM_L = OM_L
        # CDM
        self.OM_cdm = OM_tot - (OM_b + OM_v)
        # Matter density OM_m = OM_c + OM_b + OM_v
        self.OM_m = OM_tot - OM_L
        self.OM_tot = OM_tot
        self.k = np.logspace(-3, 1, 1000)
        self.h = h
        self.N_v = N_v
        self.THETA = THETA # Planck T_CMB / 2.7
        self.debug = debug

        # Quantities useful for EH 99
        self.H_0 = 100 * h # km/s/Mpc
        self.OMmh2 = self.OM_m * h**2
        self.OMbh2 = self.OM_b * h**2
        self.T_CMB = 2.7 * THETA # K
        self.f_c = self.OM_cdm / self.OM_m
        self.f_v = self.OM_v / self.OM_m
        self.f_b = self.OM_b / self.OM_m
        self.f_vb = (self.OM_b + self.OM_v) / self.OM_m
        self.f_cb = (self.OM_cdm + OM_b) / self.OM_m
            
        assert (OM_tot == 1), "Only flat cosmologies accepted! OM_tot must be 1"
        assert (self.OM_m + self.OM_L == self.OM_tot), "OM_m + OM_L must be equal to OM_tot!"
        if debug:
            self.print_cosmo_params()
    
    def clone(self):
        return Cosmology(z = self.z,
                          OM_tot = self.OM_tot,
                          OM_b = self.OM_b,
                          OM_v = self.OM_v,
                          OM_L = self.OM_L,
                          h = self.h, 
                          N_v = self.N_v,
                          THETA = self.THETA)

    def update_z(self, z):
        self.z = z
        return

    def print_cosmo_params(self):
        n = 10
        line = '------------------------------'
        print('Input cosmological parameters')
        print(line)
        params = ['z', 'Ω_cdm', 'Ω_b', 'Ω_v', 'Ω_m', 'Ω_Λ', 'Ω_tot', 'h', 'N_v', 'T_CMB']
        vals = [self.z, self.OM_cdm, self.OM_b, self.OM_v, self.OM_m, self.OM_L, self.OM_tot, self.h, self.N_v, self.THETA * 2.7]
        for i in range(len(params)):
            spaces = ' ' * (n - len(params[i]))
            print(params[i] + spaces + str(vals[i]))
        print(line)
        return   

    def g_z(self, OMm_z):
        # eq. 6 from Liddle et al. 1994
        return (5/2.0)*OMm_z/(1/(70.0) +
                 (209/140.0) * OMm_z -
                 OMm_z * OMm_z / 140.0 +
                 (OMm_z**(4/7.0)));
      
    def D(self, z = None):
        # Function D(z) is the growth factor. It returns the 
        # linear growth of perturbations, normalized
        # to 1 at z = 0.
        if z is None:
            z = self.z
        if self.OM_m == 1 and self.OM_tot == self.OM_m:
            return 1 / (1 + z)
        elif self.OM_tot == self.OM_m + self.OM_L and self.OM_tot == 1:
            OMm_z = self.OM_m * (1 + z)**3. / (self.OM_L + self.OM_m * (1 + z)**3.);
            return self.g_z(OMm_z) / self.g_z(self.OM_m) / (1.0 + z);
        else:
            raise ValueError("ERROR: no growth function for this cosmology! returning -1\n");

    def delta_c(self, z = None):
        if z is None:
            z = self.z
        # Critical density as a function of redshift z
        return 1.686/self.D(z)
    
    def mean_density(self, z = None):
        r"""
        H(z) = H_0 * E(z), where
        E(z) = (\sum_i{\Omega_{i0} * (1 + z)^n_i} )^(1/2) (8.119 and 8.120 in Carroll GR)
        => flat and MD E(z) = (1 + z)^(3/2)
        [Mean density of the universe at redshift z = 0] = \bar{ρ(z=0)}
        """
        if z is None:
            z = self.z
        G = 6.67408e-11 # m^3 / (kg s^2)
        kg_to_M_sun = 1/1.988475e30
        m_to_km = 1/1e-3
        m_to_Mpc = 1/3.2407792700054e-23
        # Calculate \rho_c(z = 0) in units of M_sun / Mpc^3
        rho0 = 3. * (self.h * 100.) ** 2 / (8. * np.pi * G) * kg_to_M_sun * m_to_km ** 2 * m_to_Mpc
        mean_rho_0 = self.OM_m * rho0 / self.h ** 2 #M_sun * h^2 / Mpc^3
        if z == 0:
            return mean_rho_0
        else:
            return mean_rho_0 #* (1+z)**3
