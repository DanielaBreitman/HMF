
import numpy as np 
import matplotlib.pyplot as plt
# To calculate runtime
from datetime import datetime

# My code
from cosmology import Cosmology
from hmf_calc import HMF

global font
font  = 15

def test_growth():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from hmf import MassFunction
    Ms = np.logspace(8,15,10000)
    Zs = np.linspace(0,15,50)
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    for z in Zs:
        module_hmf = MassFunction(z=z, Mmin = 8, Mmax = 15, transfer_model = 'BBKS', growth_model = 'Carroll1992', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                      transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        cosmo = Cosmology(z = z, OM_tot = 1.,
                OM_b = module_hmf.cosmo.Ob0,
                OM_v = 0.,
                OM_L = 1. - module_hmf.cosmo.Om0,
                h = module_hmf.cosmo.H0.value / 100., 
                THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        
        ax[0].scatter(z, cosmo.D(z), color = 'b', marker = '*')
        ax[0].scatter(z, module_hmf.growth_factor, color = 'r', marker = '.')
        ax[1].scatter(z, (cosmo.D(z) - module_hmf.growth_factor) / module_hmf.growth_factor, color = 'b', marker = '*')
    ax[0].scatter(z, cosmo.D(z), color = 'b', marker = '*',label = 'This code')
    ax[0].scatter(z, module_hmf.growth_factor, color = 'r', marker = '.', label = 'HMF')
    ax[1].axhline(0, ls = '--', color = 'r')
    ax[1].set_ylabel(r'$\frac{D_{mine} - D_{HMF} }{D_{HMF}}$', fontsize = font + 10)
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Redshift z', fontsize = font)
    ax[0].set_ylabel('Growth factor D', fontsize = font)
    plt.tight_layout()
    #plt.savefig('D.png', dpi = 300)
    plt.show()

def test_T_CDM():
    """
    ###### For this plot to be correct, must remove factor of h in denominator of q in T_CDM in transfer.py ######

    """
    from transfer import Transfer
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    from hmf import Transfer
    tr_CDM = Transfer(z = 0, transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk, 
                      transfer_params = {"use_sugiyama_baryons": True})
    print(tr_CDM.cosmo)
    cosmo = Cosmology(z = 0, OM_tot = 1.,
                OM_b = tr_CDM.cosmo.Ob0,
                OM_v = 0.,
                OM_L = 1. - tr_CDM.cosmo.Om0,
                h = tr_CDM.cosmo.H0.value / 100., 
                THETA = tr_CDM.cosmo.Tcmb0.value / 2.7, debug = True)
    from transfer import Transfer
    T_CDM = Transfer(cosmo, k = k, transfer_fnc_type = 'B86')
    ax[0].plot(k, T_CDM.transfer , label = 'My code', color = 'k')
    ax[0].plot(tr_CDM.k, np.exp(tr_CDM._unnormalised_lnT), ls = '--', color = 'r', label = 'HMF BBKS reference')
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Transfer function $T_{CDM}$', fontsize = font)
    ax[1].set_xlabel('Wavenumber k [h/Mpc]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{T_{CDM_{HMF}} - T_{CDM_{mine}} }{T_{CDM_{HMF}}}$', fontsize = font + 10)
    ax[1].plot(k, (np.exp(tr_CDM._unnormalised_lnT) - T_CDM.transfer)/np.exp(tr_CDM._unnormalised_lnT), color = 'k')
    ax[1].set_xscale('log')
    ax[1].axhline(0, ls = '--', color = 'r')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.savefig('T_CDM_Test.png', dpi = 300)
    plt.show()

def test_T_EH():
    """
    ###### For this plot to be correct, must remove factor of h in denominator of q in T_CDM in transfer.py ######

    """
    from transfer import Transfer
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from cosmology import Cosmology
    cosmo = Cosmology(z = 0, OM_tot = 1.,
                OM_b = 0.05,
                OM_v = 0.,
                OM_L = 0.,
                h = 0.5, 
                THETA = 2.73 / 2.7, debug = True)
    T_EH = Transfer(cosmo = cosmo, k = k, transfer_fnc_type = 'EH99',
                a_v = 1, B_c = 1, q_eff_is_q = True)
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    ax[0].plot(k, T_EH.transfer , label = 'My code', color = 'k')
    from hmf import Transfer, Cosmology
    from astropy.cosmology import Planck15
    tr_EH = Transfer(z = 0, transfer_model = 'EH',lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk)
    tr_EH.update(cosmo_params={'Om0': 1., 'Ob0' : 0.05, 'H0' : 50}, z=0)
    print(tr_EH.cosmo)
    ax[0].plot(tr_EH.k, np.exp(tr_EH._unnormalised_lnT), ls = '--', color = 'r', label = 'HMF EH reference')
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Transfer function $T_{EH}$', fontsize = font)
    ax[1].set_xlabel('Wavenumber k [h/Mpc]', fontsize = font)

    ax[1].set_ylabel(r'Fractional error $\frac{T_{EH_{mine}} - T_{EH_{HMF}} }{T_{EH_{HMF}}}$', fontsize = font + 10)
    ax[1].plot(k, (T_EH.transfer - np.exp(tr_EH._unnormalised_lnT))/np.exp(tr_EH._unnormalised_lnT), color = 'k')
    ax[1].set_xscale('log')
    ax[1].axhline(0, ls = '--', color = 'r')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.savefig('T_EH_Test.png', dpi = 300)
    plt.show()

def test_T_EH2():
    """
    ###### For this plot to be correct, must remove factor of h in denominator of q in T_CDM in transfer.py ######
    # Make Fig 1:

    """
    from transfer import Transfer
    lnkmin, lnkmax, dlnk = np.log(1e-3), np.log(1e2), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from cosmology import Cosmology
    cosmo = Cosmology(z = 0, OM_tot = 1.,
                OM_b = 0.05,
                OM_v = 0.2,
                OM_L = 0.,
                h = 0.5, 
                THETA = 2.725 / 2.7, debug = True)
    T_EH = Transfer(cosmo = cosmo, k = k, transfer_fnc_type = 'EH99_T_cbv',
                a_v = 1, B_c = 1, q_eff_is_q = True)
    T_CDM = Transfer(cosmo = cosmo, k = k, transfer_fnc_type = 'B86')
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    from hmf import Transfer
    tr_EH = Transfer(z = 0, transfer_model = 'EH',lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk)
    tr_EH.update(cosmo_params={'Om0': 1., 'Ob0' : 0.05, 'H0' : 50}, z=0)
    tr_CDM = Transfer(z=0, transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk, 
                      transfer_params = {"use_sugiyama_baryons": True})
    tr_CDM.update(cosmo_params={'Om0': 1., 'Ob0' : 0.05, 'H0' : 50}, z=0)
    print(tr_EH.cosmo)
    ax[0].plot(k, np.exp(tr_EH._unnormalised_lnT) / np.exp(tr_CDM._unnormalised_lnT), label = 'My code', color = 'k')
    #ax[0].plot(tr_EH.k, np.exp(tr_EH._unnormalised_lnT), ls = '--', color = 'r', label = 'HMF BBKS reference')
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('Transfer function $T_{EH}$', fontsize = font)
    ax[1].set_xlabel('Wavenumber k [h/Mpc]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{T_{EH_{mine}} - T_{EH_{HMF}} }{T_{EH_{HMF}}}$', fontsize = font + 10)
    ax[1].plot(k, (T_EH.transfer - T_CDM.transfer)/T_CDM.transfer, color = 'k')
    ax[1].set_xscale('log')
    ax[1].axhline(0, ls = '--', color = 'r')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.savefig('T_EH_Test2.png', dpi = 300)
    plt.show()

def test_MPS():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    Zs = [0, 6, 12]
    Cs = ['k','r','b']
    Cs2 = ['gray','pink','cyan']
    for z,c1,c2 in zip(Zs, Cs, Cs2):
        from hmf import Transfer
        tr_CDM = Transfer(z = z, growth_model = 'Carroll1992', transfer_model = 'BBKS', lnk_min = lnkmin, 
                          lnk_max = lnkmax, dlnk = dlnk,transfer_params = {"use_sugiyama_baryons": True})
        from mps import MPS
        P = MPS(z = z, k = k, transfer_fnc_type = 'B86',
                    OM_b = tr_CDM.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - tr_CDM.cosmo.Om0,
                    h = tr_CDM.cosmo.H0.value / 100., 
                    sigma_8 = tr_CDM.sigma_8, 
                    n = tr_CDM.n,
                    THETA = tr_CDM.cosmo.Tcmb0.value / 2.7).sigma_squared_k
        ax[0].plot(k, P, color = c1, label = 'z = ' + str(z))
        ax[0].plot(tr_CDM.k, tr_CDM.power, ls = '--', color = c2)
        ax[1].plot(k, (P - tr_CDM.power) / (tr_CDM.power), color = c1)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\sigma^2_k\: [Mpc^6]$', fontsize = font)
    ax[1].set_xlabel('Wavenumber k [h/Mpc]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\sigma^2_{k_{mine}}  - \sigma^2_{k_{HMF}}}{\sigma^2_{k_{HMF}}}$', fontsize = font + 10)
    ax[1].set_xscale('log')
    ax[1].axhline(0, ls = '--', color = 'r')
    ax[0].legend(fontsize = font)
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.tight_layout()
    plt.savefig('MPS.png', dpi = 300)
    plt.show()

def test_sigma_M():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    from hmf import MassFunction
    Zs = [0, 6, 12]
    Cs = ['k','r','b']
    Cs2 = ['gray','pink','cyan']
    for z,c1,c2 in zip(Zs, Cs, Cs2):
        module_hmf = MassFunction(z=z, Mmin = 8, Mmax = 15, growth_model = 'Carroll1992', transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                          transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        cosmo = Cosmology(z = z, OM_tot = 1.,
                    OM_b = module_hmf.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - module_hmf.cosmo.Om0,
                    h = module_hmf.cosmo.H0.value / 100., 
                    THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        halomf = HMF(z = z, k = k, M = module_hmf.m, cosmo = cosmo, transfer_fnc_type = 'B86').sig_squared_M(z=z)**0.5
        ax[0].plot(module_hmf.m, halomf, color = c1, label = 'z = ' + str(z))
        y = module_hmf.sigma
        ax[0].plot(module_hmf.m, y, ls = '--', color = c2)
        ax[1].plot(module_hmf.m, (halomf - y)/y, color = c1)
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlim(1e8,1e15)
    ax[0].set_ylabel(r'$\sigma_M [Mpc^3] $', fontsize = font)
    ax[1].set_xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\sigma_{M_{HMF}}  - \sigma_{M_{mine}}}{\sigma_{M_{HMF}}}$', fontsize = font + 10)
    ax[1].set_xscale('log')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.tight_layout()
    plt.savefig('sig_M.png', dpi = 300)
    plt.show()

def test_derivative():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from hmf import MassFunction
    Ms = np.logspace(8,15,10000)
    module_hmf = MassFunction(z=0, Mmin = 8, Mmax = 15, transfer_model = 'BBKS', growth_model = 'Carroll1992', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                      transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
    y = module_hmf.sigma* abs(module_hmf._dlnsdlnm)/module_hmf.m
    #print(abs(module_hmf._dlnsdlnm)/module_hmf.m)
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    cosmo = Cosmology(z = 0, OM_tot = 1.,
                OM_b = module_hmf.cosmo.Ob0,
                OM_v = 0.,
                OM_L = 1. - module_hmf.cosmo.Om0,
                h = module_hmf.cosmo.H0.value / 100., 
                THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
    halomf = HMF(z = 0, M = module_hmf.m, cosmo = cosmo, method = 'numerical', transfer_fnc_type = 'B86').dsig_M_dM
    ax[0].plot(module_hmf.m, halomf, color = 'b',label = 'numerical')
    ax[1].plot(module_hmf.m, (halomf - y)/y, color = 'b')
    halomf = HMF(z = 0, M = module_hmf.m, cosmo = cosmo, method = 'analytical', transfer_fnc_type = 'B86').dsig_M_dM
    ax[0].plot(module_hmf.m, halomf, label = 'analytical', color = 'k',)
    ax[0].plot(module_hmf.m, y, ls = '--', color = 'r', label = 'HMF BBKS reference')
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    ax[0].set_xlim(1e8,1e15)
    ax[0].set_ylabel(r'$\frac{d\sigma_M}{dM}$', fontsize = font + 10)
    ax[1].set_xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\frac{d\sigma_M}{dM}_{HMF}  - \frac{d\sigma_M}{dM}_{mine}}{\frac{d\sigma_M}{dM}_{HMF}}$', fontsize = font + 10)
    ax[1].plot(module_hmf.m, (halomf - y)/y, color = 'k')
    ax[1].set_xscale('log')
    plt.tight_layout()
    plt.savefig('dsig_M_dM.png', dpi = 300)
    plt.show()

def test_derivative_vs_z():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    from hmf import MassFunction
    Zs = [0, 6, 12]
    Cs = ['k','r','b']
    Cs2 = ['gray','pink','cyan']
    for z,c1,c2 in zip(Zs, Cs, Cs2):
        module_hmf = MassFunction(z = z, Mmin = 8, Mmax = 15, transfer_model = 'BBKS', growth_model = 'Carroll1992', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                          transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        y = module_hmf.sigma* abs(module_hmf._dlnsdlnm)/module_hmf.m
        cosmo = Cosmology(z = z, OM_tot = 1.,
                    OM_b = module_hmf.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - module_hmf.cosmo.Om0,
                    h = module_hmf.cosmo.H0.value / 100., 
                    THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        halomf = HMF(M = module_hmf.m, cosmo = cosmo, transfer_fnc_type = 'B86').derivative(z=z)
        ax[0].plot(module_hmf.m, halomf, color = c1, label = 'z = ' + str(z))
        ax[1].plot(module_hmf.m, (halomf - y)/y, color = c1)
        ax[0].plot(module_hmf.m, y, ls = '--', color = c2)
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    ax[0].set_xlim(1e8,1e15)
    ax[0].set_ylabel(r'$\frac{d\sigma_M}{dM}$', fontsize = font+10)
    ax[1].set_xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\frac{d\sigma_M}{dM}_{HMF}  - \frac{d\sigma_M}{dM}_{mine}}{\frac{d\sigma_M}{dM}_{HMF}}$', fontsize = font + 10)
    ax[1].set_xscale('log')
    plt.tight_layout()
    plt.savefig('dsig_M_dM_vs_z.png', dpi = 300)
    plt.show()

def test_HMF():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from hmf import MassFunction
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    from hmf import MassFunction
    Zs = [0, 6, 12]
    Cs = ['k','r','b']
    Cs2 = ['gray','pink','cyan']
    for z,c1,c2 in zip(Zs, Cs, Cs2):
        module_hmf = MassFunction(z=z, Mmin = 8, Mmax = 15, growth_model = 'Carroll1992', transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                          transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        cosmo = Cosmology(z = z, OM_tot = 1.,
                    OM_b = module_hmf.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - module_hmf.cosmo.Om0,
                    h = module_hmf.cosmo.H0.value / 100., 
                    THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        halomf = HMF(z = z, M = module_hmf.m, cosmo = cosmo, transfer_fnc_type = 'B86').dn_dM
        ax[0].plot(module_hmf.m, halomf, color = c1, label = 'z = ' + str(z))
        ax[0].plot(module_hmf.m, module_hmf.dndm, ls = '--', color = c2)
        ax[1].plot(module_hmf.m, (halomf - module_hmf.dndm)/module_hmf.dndm, color = c1)
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    # unit is [M_\odot^{-1} Mpc^{-3} h^4]
    ax[0].set_ylabel(r'$\frac{dn}{dM}$', fontsize = font + 10)
    ax[1].set_xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\frac{dn}{dM}_{HMF}  - \frac{dn}{dM}_{mine}}{\frac{dn}{dM}_{HMF}}$', fontsize = font + 10)
    ax[1].set_xscale('log')
    ax[0].set_xlim(1e8,1e15)
    #ax[0].set_ylim(1e-4,1e2)
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.tight_layout()
    plt.savefig('dn_dM.png', dpi = 300)
    plt.show()

def test_HMF2():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from hmf import MassFunction
    fig, ax = plt.subplots(2, 1, figsize = (12,8), sharex = True)
    fig.subplots_adjust(wspace=0, hspace=0)
    from hmf import MassFunction
    runtimes = []
    Zs = [6, 9, 12, 15]
    Cs = ['k','r','b', 'g']
    Cs2 = ['gray','pink','cyan', 'lime']
    for z,c1,c2 in zip(Zs, Cs, Cs2):
        start = datetime.now()
        module_hmf = MassFunction(z=z, Mmin = 8, Mmax = 12, growth_model = 'Carroll1992', transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                          transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        end = datetime.now()
        hmfruntime = end - start
        print('HMF runtime:', hmfruntime)
        start = datetime.now()
        cosmo = Cosmology(z = z, OM_tot = 1.,
                    OM_b = module_hmf.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - module_hmf.cosmo.Om0,
                    h = module_hmf.cosmo.H0.value / 100., 
                    THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        halomf = module_hmf.m * HMF(z = z, M = module_hmf.m, cosmo = cosmo, transfer_fnc_type = 'B86').dn_dM
        end = datetime.now()
        myruntime = end - start
        print('My runtime:', myruntime)
        runtimes.append(hmfruntime/myruntime)
        ax[0].plot(module_hmf.m, halomf, color = c1, label = 'z = ' + str(z))
        ax[0].plot(module_hmf.m,module_hmf.m * module_hmf.dndm, ls = '--', color = c2)
        ax[1].plot(module_hmf.m, (halomf - module_hmf.m* module_hmf.dndm)/(module_hmf.m*module_hmf.dndm), color = c1)
    print('My code runs ' + str(round(np.mean(runtimes),2)) + ' times faster than HMF module on average.')
    print('Number of mass points: 1000')
    print('Number of k points: ', len(module_hmf.m))
    ax[0].legend(fontsize = font)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    # unit is [M_\odot^{-1} Mpc^{-3} h^4]
    ax[0].set_ylabel(r'$M \cdot \frac{dn}{dM}$', fontsize = font + 10)
    ax[1].set_xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)

    ax[1].set_ylabel(r'$\frac{\frac{dn}{dM}_{HMF}  - \frac{dn}{dM}_{mine}}{\frac{dn}{dM}_{HMF}}$', fontsize = font + 10)
    ax[1].set_xscale('log')
    ax[0].set_xlim(1e8,1e12)
    ax[0].set_ylim(1e-4,1e2)
    ax[1].set_ylim(-1e-2,1e-2)
    plt.setp(ax[0].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_xticklabels(), fontsize = font - 1)
    plt.setp(ax[0].get_yticklabels(), fontsize = font - 1)
    plt.setp(ax[1].get_yticklabels(), fontsize = font - 1)
    plt.tight_layout()
    plt.savefig('Mdn_dM.png', dpi = 300)
    plt.show()

def test_runtime(num = 1000):
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    from hmf import MassFunction
    hmf_runtimes = np.zeros(num)
    my_runtimes = np.zeros(num)
    for i in range(num):
        start = datetime.now()
        module_hmf = MassFunction(z=0, Mmin = 8, Mmax = 12, growth_model = 'Carroll1992', transfer_model = 'BBKS', lnk_min = lnkmin, lnk_max = lnkmax, dlnk = dlnk,
                          transfer_params = {"use_sugiyama_baryons": True}, hmf_model = 'PS')
        end = datetime.now()
        hmf_runtimes[i] = (end - start).total_seconds()
        start = datetime.now()
        cosmo = Cosmology(z = 0, OM_tot = 1.,
                    OM_b = module_hmf.cosmo.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - module_hmf.cosmo.Om0,
                    h = module_hmf.cosmo.H0.value / 100., 
                    THETA = module_hmf.cosmo.Tcmb0.value / 2.7)
        halomf = module_hmf.m * HMF(M = module_hmf.m, cosmo = cosmo, transfer_fnc_type = 'B86').dn_dM
        end = datetime.now()
        my_runtimes[i] = (end - start).total_seconds()
    colours = ['r', 'k']
    from matplotlib import colors
    fc = [colors.to_rgb(c) for c in colours]
    colours = [c + (0.2,) for c in fc]
    plt.hist(my_runtimes, bins = 30, histtype = 'stepfilled', color = colours[0], edgecolor = 'r', lw = 1, label = 'This code')
    plt.hist(hmf_runtimes, bins = 30, histtype = 'stepfilled', color = colours[1], edgecolor = 'k', lw = 1, label = 'HMF')
    plt.legend(fontsize = font)
    plt.xlabel('Runtime (s)', fontsize = font)
    plt.ylabel('Count', fontsize = font)
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    plt.xlim(0,0.3)
    plt.tight_layout()
    plt.savefig('runtime.png', dpi = 300)
    plt.show()

def test_HAM():
    lnkmin, lnkmax, dlnk = np.log(1e-8), np.log(1e5), 0.01
    k = np.exp(np.arange(lnkmin, lnkmax, dlnk))
    Zs = np.array([4, 5, 6, 7, 8, 10])
    Cs = ['k','r','b', 'g', 'lime', 'cyan']
    mass = np.logspace(8, 12, 500)
    Mhmin = []
    from astropy.cosmology import Planck15 as P15
    from hmf_calc import HMF
    for z, c1 in zip(Zs, Cs):
        cosmo = Cosmology(z = z, OM_tot = 1.,
                    OM_b = P15.Ob0,
                    OM_v = 0.,
                    OM_L = 1. - P15.Om0,
                    h = P15.H0.value / 100., 
                    THETA = P15.Tcmb0.value / 2.7)
        halomf = HMF(z = z, M = mass, cosmo = cosmo, transfer_fnc_type = 'B86').dn_dM
        mask = np.argmin(abs(mass * halomf - 1e-4))
        Mhmin.append(mass[mask])
        plt.plot(mass, mass * halomf, color = c1, label = 'z = ' + str(z))
    Mhmin = np.array(Mhmin)
    plt.legend(fontsize = font)
    plt.xscale('log')
    plt.yscale('log')
    # unit is [M_\odot^{-1} Mpc^{-3} h^4]
    plt.ylabel(r'$M \cdot \frac{dn}{dM}$', fontsize = font + 10)
    plt.xlabel(r'Mass [$M_\odot$ / h]', fontsize = font)
    plt.xlim(1e8,1e12)
    plt.ylim(1e-4,1e2)
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    plt.tight_layout()
    plt.savefig('Checkdn_dM.png', dpi = 300)
    plt.show()

    plt.scatter(Zs, Mhmin, color = 'k')
    # We will fit a (1+z)^(-1.5) curve to this
    from scipy.optimize import curve_fit
    def fit_func(z, norm):
        return norm * (1 + z) ** (-3./2.)
    def fit_func2(z, norm, n):
        return norm * (1 + z) ** (n)
    popt, pcov = curve_fit(fit_func, Zs, Mhmin, p0 = [Mhmin[-1]])
    plt.plot(Zs, fit_func(Zs, *popt), color = 'r', ls = '--', label = r'{:.1e}'.format(popt[0]) + r'(1+z)$^{-1.5}$')
    popt, pcov = curve_fit(fit_func2, Zs, Mhmin, p0 = [Mhmin[-1],-1.5])
    plt.plot(Zs, fit_func2(Zs, *popt), color = 'b', ls = '--', label = r'{:.1e}'.format(popt[0]) + r'(1+z)$^{'+ str(round(popt[-1],2)) + '}$')
    plt.xlabel(r'Redshift z', fontsize = font)
    plt.ylabel(r'M$_{min}$ [$M_\odot$ / h]', fontsize = font)
    plt.legend(fontsize = font)
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    plt.tight_layout()
    plt.savefig('CheckMhmin.png', dpi = 300)
    plt.show()

#test_growth()
#test_T_CDM()
#test_T_EH()
#test_T_EH2()
#test_MPS()
#test_sigma_M()
#test_derivative_vs_z()
#test_derivative()
#test_HMF()
test_HMF2()
#test_runtime(1000)
#test_HAM()
