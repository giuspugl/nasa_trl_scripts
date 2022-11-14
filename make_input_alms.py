import pysm3
import pylab as pl 
import healpy as hp
import numpy as np 
from astropy import units as   u 
from astropy import constants as   const 
import toml 
import pysm3.units as u3
import warnings
warnings.filterwarnings("ignore")
from so_pysm_models import extragalactic 

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


hwdir ='/global/cscratch1/sd/giuspugl/nasa_trl_demo_sims/hardware_config/' 
dic_lb = toml.load(f'{hwdir}/litebirdms_hardware.toml')
detectors  = [b for b in dic_lb['detectors'].keys()   ]  

lmax=1024
mmax=1024 
 

nside=512

galsky = pysm3.Sky(nside=nside, preset_strings=["s5","d9","a1","f1","co1"],output_unit='uK_RJ')
tsz =extragalactic.WebSkySZ(nside=nside, sz_type='thermal' )
ksz =extragalactic.WebSkySZ(nside=nside, sz_type='kinetic' )
cib = extragalactic.WebSkyCIB(nside=nside)
lens_cmb =hp.read_map(f"/global/cfs/cdirs/litebird/simulations/maps/post_ptep_inputs_20220522/websky_extragal/websky_lensed_cmb/CMB/sims_00.fits", field=None )
#radio = hp.read_map 
lens_cmb =hp.ud_grade(lens_cmb, nside_out=nside  ) *u3.uK_CMB 


out_dir ="/global/cscratch1/sd/giuspugl/nasa_trl_demo_sims/sky_inputs_alm/" 



for det in detectors : 
    bandstring = (dic_lb['detectors'] [det] ['band'] )
    fwhm = dic_lb['detectors'] [det] ['fwhm'] *u.arcmin 
    b0 = dic_lb['bands'][bandstring]['center'] *u.GHz
    bw = dic_lb['bands'][bandstring]['bandwidth'] *u.GHz 
    band = pl.linspace(b0-bw/2, b0+bw/2, 16) 
    #read bandpasses from disc 
    bandpass = np.ones_like(band.value)
    print(det, b0,bw ,fwhm ) 
 
    galaxy  = galsky.get_emission(freq=band, weights= bandpass )
    thermo_sz=  tsz.get_emission(freqs=band,weights=bandpass) 

    kin_sz=  ksz.get_emission(freqs=band,weights=bandpass)   
    cibmap=  cib.get_emission(freqs=band,weights=bandpass)
    extra_gal = thermo_sz +kin_sz + cibmap #+radio 
    conversion_factor = Krj2Kcmb(  nu=band, bandpass=bandpass ).value *u3.uK_CMB
    
    print(conversion_factor) 
    coadd = lens_cmb +conversion_factor *(extra_gal.value + galaxy.value  )
    alms = hp.map2alm(maps =coadd ,lmax =lmax, mmax=mmax ) 
    
    alms = hp.smoothalm( alms=alms ,fwhm=fwhm.to(u.rad).value  ,mmax=mmax ) 
    almT = np.zeros_like(alms ) 
    almT[0] = alms[0] 
    almEB = np.zeros_like(alms ) 
    almEB [1:] = alms[1:] 
    almBE = np.zeros_like(alms ) 
    almBE [2] = -alms[1] 
    almBE [1] = alms[2]
    
    hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_T.fits" , alms =almT , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
    hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_EB.fits" , alms =almEB , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
    hp.write_alm(filename=f"{out_dir}/sky_alm_{det}_BE.fits" , alms =almBE, lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
    
    
    
    break 