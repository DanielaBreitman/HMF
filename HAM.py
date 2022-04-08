import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import simpson
from astropy.cosmology import Planck15 as cosmoP15

from hmf_calc import HMF
from cosmology import Cosmology

from astropy import units as u
from astropy.coordinates import SkyCoord, Distance

global font
font = 15
def read_galaxy_data():
    # Make data table that contains RA [deg], Dec [deg], AB magnitude [mag], and photometric z estimate [ ]
    fpath = '/home/dani/PhD/Structure_Formation/Midterm/python_version/apj508910t3_mrt.txt'
    RA = [] # deg
    Dec = [] # deg
    m_obs = [] # mag
    z = []
    survey_num = []
    with open(fpath,"r") as data:
        for line in data:
            if '#' in line:
                continue
            else:
                this_line = np.array(line.split(' '))
                this_line = this_line[this_line != '']
                # Convert RA from hms to deg:
                c = SkyCoord(ra = this_line[1] + 'h' + this_line[2] + 'm' + this_line[3] + 's', 
                            dec = this_line[4] + 'd' + this_line[5] + 'm' + this_line[6] + 's', frame = 'icrs')
                RA.append(c.ra.degree)
                Dec.append(c.dec.degree)
                m_obs.append(float(this_line[7]))
                if '*' in this_line[8]:
                    z.append(float(this_line[8][:-1]))
                else:
                    z.append(float(this_line[8]))
                survey_num.append(int(this_line[9]))
    M_1600 = m2M(np.array(m_obs), np.array(z))
    return {'RA' : np.array(RA), 'Dec': np.array(Dec), 'm_obs' : np.array(m_obs), 
            'z' : np.array(z), 'M_1600_AB' : M_1600, 'survey_num' : np.array(survey_num)}

def data_plot(data):
    for i in range(10):
        mask = data['survey_num'] == i + 1
        print('Number of galaxies in survey: ', sum(mask))
        fig0 = plt.scatter(data['RA'][mask], data['Dec'][mask], 
                c = data['M_1600_AB'][mask], s = data['z'][mask] ** 2, cmap = 'plasma', alpha = 0.7)
        plt.ylabel('Dec (deg)', fontsize = font)
        plt.xlabel('RA (deg)', fontsize = font)
        cbar = plt.colorbar(fig0)
        cbar.set_label(r'$M_{1600,AB}$', fontsize = font)
        plt.xticks(fontsize = font - 5)
        plt.yticks(fontsize = font)
        plt.tight_layout()

        #plt.savefig('RA-Dec' + str(i + 1) + '.png', dpi = 400)
        plt.show()
    return
    
def read_LF():
    fpath = '/home/dani/PhD/Structure_Formation/Midterm/python_version/UV_LFs.txt'
    M_1600_AB = [] # mag
    LF = [] # Mpc^{-3} mag{-1}
    LF_err = [] # Mpc^{-3} mag{-1}
    z = []
    with open(fpath,"r") as data:
        for line in data:
            this_line = line.split('\t')
            M_1600_AB.append(float(this_line[0].replace("−", "-")))
            LF.append(float(this_line[1]))
            if '-' not in this_line[2]:
                LF_err.append(float(this_line[2]))
            else:
                LF_err.append(np.nan)
            z.append(float(this_line[3].split('\n')[0]))
    return {'M_1600_AB': np.array(M_1600_AB), 'LF': np.array(LF), 'LF_err' : np.array(LF_err), 'z': np.array(z)}


def find_lb(int_val, cosmo, trial_lbs = None, error = 1e-3):
    """
    Assumptions:
        Integral upper bound = very large -->infinity
        Integral lower bound > 0.
        Integral > 0.
    Find integral lower bound given integrand function f
    and the value that we want the integral to have, int_val.

    i.e. Find a, where int_val = \int_a^{b -->\inf} f(x) dx with precision 'error'.
    """
    if trial_lbs is None:
        start, end = -50, 50
        trial_lbs = np.logspace(start,end,50)
    lb_quality_measure = np.zeros(len(trial_lbs))
    for i in range(len(trial_lbs)):
        x = np.logspace(np.log10(trial_lbs[i]), 60, 500)
        hmf = HMF(M = x, cosmo = cosmo, transfer_fnc_type = 'L96').calc_hmf()
        this_int = simpson(hmf, x)
        #print(this_int, trial_lbs[i])
        lb_quality_measure[i] = (int_val - this_int)/int_val
    closest_lb = np.nanargmin(abs(lb_quality_measure))
    print('Delta integrals / LF integral = ', lb_quality_measure[closest_lb])
    # Is closest lb too big or too small?
    if lb_quality_measure[closest_lb] > error/2.:
        # closest_lb is too big
        new_trial_lbs = np.logspace(np.log10(trial_lbs[closest_lb - 1]), np.log10(trial_lbs[closest_lb]), 50)
        return find_lb(int_val, cosmo, new_trial_lbs, error = error)
    elif lb_quality_measure[closest_lb] < -0.5 * error:
        # closest_lb is too small
        new_trial_lbs = np.logspace(np.log10(trial_lbs[closest_lb]), np.log10(trial_lbs[closest_lb + 1]), 50)
        return find_lb(int_val, cosmo, new_trial_lbs, error = error)
    else:
        return trial_lbs[closest_lb], lb_quality_measure[closest_lb]

def m2M(m, z):
    """
    Convert observed magnitude m_obs to absolute magnitude
    M_1600_AB using the photometric redshift z, assuming Planck15
    cosmology.

    Parameters
    ----------
    m : np.ndarray or float
        Observed magnitude [mag]
    z : np.ndarray or float
        Photometric redshift [ ]
    
    Returns
    -------
    Absolute magnitude M
    """
    # m - M = 5 log_10 (d_L / 10 pc)
    # Uses default Planck cosmology 
    d_L = Distance(z = z, cosmology = cosmoP15).pc
    return m - 5. * np.log10(d_L/10.) 

def plot_LFs(data):
    import matplotlib.pylab as pl
    Zs = np.unique(data['z'])
    M = []
    LF = []
    colors = pl.cm.viridis(np.linspace(0,1,len(Zs)))
    for i in range(len(Zs)):
        mask = data['z'] == Zs[i]
        LF.append(data['LF'][mask])
        M.append(data['M_1600_AB'][mask])
        plt.errorbar(data['M_1600_AB'][mask], data['LF'][mask], color = colors[i], yerr = data['LF_err'][mask], label = 'z = ' + str(Zs[i]))
    # Show that the survey is magnitude-limited
    ratios3 = []
    ratios2 = []
    new_Zs = np.linspace(4,10,30)
    for i in range(len(new_Zs)):
        d_L1 = Distance(z = new_Zs[0], cosmology = cosmoP15).pc
        d_L2 = Distance(z = new_Zs[i], cosmology = cosmoP15).pc
        ratios3.append((d_L1/d_L2)**3)
        ratios2.append((d_L1/d_L2)**2)
    ratios3 = np.array(ratios3)
    ratios2 = np.array(ratios2)
    def Ms(z, mlim = 31.5):
         return mlim - 5. * np.log10(Distance(z = z, cosmology = cosmoP15).pc / 10.)
    colors = pl.cm.viridis(np.linspace(0,1,len(new_Zs)))
    plt.scatter(M[0][-1] /(Ms(4)/Ms(new_Zs)), LF[0][-1] * np.ones(len(ratios2)), marker = 'o', color = colors)
    plt.scatter(M[0][-1] /(Ms(4)/Ms(new_Zs)), LF[0][-1] * ratios3, marker = 'o', color = colors)
    plt.xlabel(r'$M_{1600,AB}$ [mag$^{-1}$]', fontsize = font)
    plt.ylabel(r'$\phi$ [Mpc$^{-3}$ mag$^{-1}]$', fontsize = font)
    plt.yscale('log')
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    plt.legend(fontsize = font)
    plt.tight_layout()
    plt.savefig('B+14_LFs.png', dpi = 300)
    plt.show()

def HAM(UV_LF_data):
    Zs = np.unique(UV_LF_data['z'])
    Mhmin = np.zeros(len(Zs))
    #for i in range(len(Zs)):
        # Get UV data for this redshift:
    #    mask = UV_LF_data['z'] == Zs[i]
        # Integrate UV LF:
    #    UV_LF_int = simpson(UV_LF_data['LF'][mask], UV_LF_data['M_1600_AB'][mask])
    #    print(UV_LF_int, Zs[i])
        # Initialize HMF object
    #    cosmo = Cosmology(z = Zs[i], OM_tot = 1.,
    #                    OM_b = cosmoP15.Ob0,
    #                    OM_v = 0.,
    #                    OM_L = 1. - cosmoP15.Om0,
    #                    h = cosmoP15.H0.value / 100.,
    #                    THETA = cosmoP15.Tcmb0.value / 2.7)
        # Run this function to recalculate dn/dM
        # Match abundances:
    #    M_h_min, delta = find_lb(UV_LF_int, cosmo, error = 1e-3)
    #    Mhmin[i] = M_h_min
    Mhmin = [2.67543106e+10, 2.40844856e+10, 1.79744578e+10, 1.18238469e+10, 1.34579621e+10, 1.34278935e+10]
    plt.scatter(Zs, Mhmin, color = 'k')
    # We will fit a (1+z)^(-1.5) curve to this
    from scipy.optimize import curve_fit
    def fit_func(z, norm):
        return norm * (1 + z) ** (-3./2.)
    def fit_func2(z, norm, n):
        return norm * (1 + z) ** (n)
    def M_h_T_vir(z, T_vir):
        Om0, Omz = cosmoP15.Om0, cosmoP15.Om(z)
        mu = 0.6
        # Unit is M_sun
        return 1e8 * (T_vir/1e4 * (mu / 0.6 * (1 + z) / 10. * Om0 / 0.3 / Omz)**(-1))**(3/2) / (cosmoP15.H0.value / 100.)
    popt, pcov = curve_fit(fit_func, Zs, Mhmin, p0 = [Mhmin[-1]])
    #plt.plot(Zs, fit_func(Zs, *popt), color = 'r', ls = '--', label = r'{:.1e}'.format(popt[0]) + r'(1+z)$^{-1.5}$')
    popt, pcov = curve_fit(fit_func2, Zs, Mhmin, p0 = [Mhmin[-1],-1.5])
    #plt.plot(Zs, fit_func2(Zs, *popt), color = 'b', ls = '--', label = r'{:.1e}'.format(popt[0]) + r'(1+z)$^{'+ str(round(popt[-1],2)) + '}$')
    popt, pcov = curve_fit(M_h_T_vir, Zs[:-2], Mhmin[:-2], p0 = [1e5])
    plt.plot(Zs, M_h_T_vir(Zs, popt[0]), color = 'r', ls = '--', label = r'$T_{vir}$ = ' + r'{:.1e}'.format(popt[0]) + r' K')
    plt.xlabel('Redshift z', fontsize = font)
    plt.yscale('log')
    plt.legend(fontsize = font)
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    plt.ylabel(r'$M_{h_{min}}$ [M$_\odot$/h]', fontsize = font)
    plt.tight_layout()
    plt.savefig('B+14_Mhmin_vs_z.png', dpi = 300)
    plt.show()


#galaxy_data = read_galaxy_data()
# Plot RA vs Dec vs M_1600 and z for each survey
#data_plot(galaxy_data)
# Try calculating LF on my own (this is wrong)
# OR we can read the LFs directly from Bouwens+14 Table 5 data
data = read_LF()
plot_LFs(data)
HAM(data)
