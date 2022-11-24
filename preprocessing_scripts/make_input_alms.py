import pysm3
import pylab as pl 
import healpy as hp
import numpy as np 
from astropy import units as   u 
from astropy import constants as   const 
import toml 
import pysm3.units as u3
import warnings
import glob 
import time 

warnings.filterwarnings("ignore")

def b(nu):
    """
    estimate dI/dT_CMB as defined in eq. 8 of Planck 2013 IX
    """
    nu0 = 56.8 * u.GHz
    x = nu / nu0
    Tcmb = 2.7255 * u.K
    bb = (
        2
        * const.h
        * nu ** 3
        / const.c ** 2
        / Tcmb
        * (x * np.exp(x) / (np.exp(x) - 1) ** 2)
    )
    return bb.to(u.W / u.m ** 2 / u.K / u.Hz)

def b_rj(nu):
    """
    estimate dI/dT_RJ as defined in eq. 30 of Planck 2013 IX
    """
    bb = 2 * const.k_B * nu ** 2 / const.c ** 2
    return bb.to(u.W / u.m ** 2 / u.K / u.Hz)

def brightness2Tcmb(nu, bandpass=None):
    """
    Convert from Jy/srad  to K_CMB
    """
    try:
        nu.unit
    except AttributeError:
        nu *= u.GHz

    if bandpass is None:
        integrand = b(nu)
        conversion_factor = 1 / integrand
    else:
        bandpass /= u.GHz
        integrand = b(nu) * bandpass
        conversion_factor = 1 / np.trapz(integrand, x=(nu))
    return conversion_factor

def brightness2Trj(nu, bandpass=None):
    """
    Convert from Jy/srad to K_RJ
    """
    try:
        nu.unit
    except AttributeError:
        nu *= u.GHz

    if bandpass is None:
        integrand = b_rj(nu)
        conversion_factor = 1 / integrand
    else:
        bandpass /= u.GHz
        integrand = b_rj(nu) * bandpass
        conversion_factor = 1 / np.trapz(integrand, x=(nu))
    return conversion_factor

def Krj2Kcmb(nu, Trj=1.0, bandpass=None):
    """
    Convert antenna temperature ( Rayleigh-Jeans) into the physical one
    """
    return Trj / Kcmb2Krj(
        nu=nu, Tcmb=1, bandpass=bandpass
    )  # (x ** 2 * np.exp(x) / (np.exp(x) - 1) ** 2)

def Kcmb2Krj(nu, Tcmb=1.0, bandpass=None):
    return (
        brightness2Trj(nu=nu, bandpass=bandpass)
        / brightness2Tcmb(nu=nu, bandpass=bandpass)
        * Tcmb
    )



sim_dir = '/global/cfs/cdirs/litebird/simulations/TOAST_nasa_trl_sims/'
hwdir =f'{sim_dir}/hardware/' 
out_dir =f'{sim_dir}/sky_inputs_alm/' 
temp_dir =f'{sim_dir}/sky_templates/' 
bpass_dir=f'{sim_dir}/det_bandpasses/' 

hwfiles= glob.glob(f'{hwdir}/*toml.gz') 
nside=512
lmax=1024 ; mmax=20

for hwfile in hwfiles: 
    dic_lb = toml.load(hwfile)
    
    bandstring = [ k for k in dic_lb['bands'].keys()  ][0]
    fwhm = dic_lb['bands'][bandstring]['fwhm'] *u.arcmin 
    b0 = dic_lb['bands'][bandstring]['center'] *u.GHz
    bw = dic_lb['bands'][bandstring]['bandwidth'] *u.GHz 
    band = pl.linspace(b0-bw/2, b0+bw/2, 16) 
    #read bandpasses from disc 
    
    bandpass = np.ones_like(band.value)
    
    print( bandstring  )
    #low complexity foregrounds 
    # https://galsci.github.io/blog/2022/common-fiducial-sky/
    sky = pysm3.Sky(nside=nside, preset_strings=["d9","s4","f1","a1","co1","c4","cib1", "tsz1", "ksz1", "rg1" ],output_unit='K_CMB')
    skyT =sky.get_emission(freq=band, weights= bandpass )
    skyT = hp.smoothing(skyT[0], lmax=lmax , fwhm = fwhm.to(u.rad).value ) 
    
    hp.write_map(f"{temp_dir}/template_map_T_{bandstring}_top-hat_bpass_K_CMB.fits", skyT, column_units='K_CMB' )
    
    detectors = (list(dic_lb['detectors'] .keys()  ))
    bpasses = pl.load(f"{bpass_dir}/{bandstring}_cheby.npz")
    for det in detectors :
        start= time.perf_counter() 
        
        signals = sky.get_emission(freq=bpasses['freq_ghz'] , 
                                   weights= bpasses[det]  ) 
        alms = hp.map2alm(maps =signals ,lmax =lmax, mmax=mmax ) 
        ### to avoid any issue with fwhm i consider alms convolved with 10arcmin <17.8 arcmin of the highest reso LB channel 
        alms = hp.smoothalm( alms=alms ,fwhm= pl.radians(10/60.) ,lmax=lmax ,mmax=mmax ) 
        almT = np.zeros_like(alms ) 
        almT[0] = alms[0] 
        almEB = np.zeros_like(alms ) 
        almEB [1:] = alms[1:] 
        almBE = np.zeros_like(alms ) 
        almBE [2] = -alms[1] 
        almBE [1] = alms[2]

        hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_T_K_CMB.fits" , alms =almT , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
        hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_EB_K_CMB.fits" , alms =almEB , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
        hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_BE_K_CMB.fits" , alms =almBE, lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
        end= time.perf_counter() 
        print(end-start)
        
    break
    

 





