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
import os 
import mpi4py 
warnings.filterwarnings("ignore")
from mpi4py import MPI 
comm    = MPI.COMM_WORLD
rank =comm.Get_rank()
nprocs =comm.Get_size()

sim_dir = '/global/cfs/cdirs/litebird/simulations/TOAST_nasa_trl_sims/'
hwdir =f'{sim_dir}/hardware/' 
out_dir =f'{sim_dir}/sky_inputs_alm/' 
temp_dir =f'{sim_dir}/sky_templates/' 
bpass_dir=f'{sim_dir}/det_bandpasses/' 
hwfiles= glob.glob(f'{hwdir}/*toml.gz') 
arr = np.arange(len(hwfiles), dtype=int) 
loc_indices = np.array_split(arr, nprocs ) [rank]

nside=512
lmax=1024 ; mmax=1024 ##that's required by conviqt 

for indx in  loc_indices  : 
    dic_lb = toml.load(hwfiles[indx ])
    
    bandstring = [ k for k in dic_lb['bands'].keys()  ][0]
    fwhm = dic_lb['bands'][bandstring]['fwhm'] *u.arcmin 
    b0 = dic_lb['bands'][bandstring]['center'] *u.GHz
    bw = dic_lb['bands'][bandstring]['bandwidth'] *u.GHz 
    band = pl.linspace(b0-bw/2, b0+bw/2, 16) 
    #read bandpasses from disc 
    
    bandpass = np.ones_like(band.value)
    
    print( bandstring  )
    os.makedirs(f"{out_dir}/{bandstring}", exist_ok=True )
    
    #low complexity foregrounds 
    # https://galsci.github.io/blog/2022/common-fiducial-sky/   
    sky = pysm3.Sky(nside=nside, preset_strings=["d9","s4","f1","a1","co1","c4","cib1", "tsz1", "ksz1", "rg1" ],output_unit='K_CMB')
    
    if not os.path.exists( f"{temp_dir}/template_map_T_{bandstring}_top-hat_bpass_K_CMB.fits"): 
        skyT =sky.get_emission(freq=band, weights= bandpass )
        skyT = hp.smoothing(skyT[0], lmax=lmax , fwhm = fwhm.to(u.rad).value ) 
        hp.write_map(f"{temp_dir}/template_map_T_{bandstring}_top-hat_bpass_K_CMB.fits", skyT, column_units='K'  )
    else:
        print("skipping template" )
        
    detectors = (list(dic_lb['detectors'] .keys()  ))
    bpasses = pl.load(f"{bpass_dir}/{bandstring}_cheby.npz")
    for idet , det in enumerate(detectors) :
        start= time.perf_counter() 
        
        almTfile =f"{out_dir}/{bandstring}/sky_alm_{det}_T.fits"
        almEBfile =f"{out_dir}/{bandstring}/sky_alm_{det}_EB.fits"
        almBEfile =f"{out_dir}/{bandstring}/sky_alm_{det}_BE.fits"
        if  (os.path.exists(almTfile) and  os.path.exists(almEBfile) and os.path.exists(almBEfile)) :
            print(f"skipping {det}" )
            continue 
        

        signals = sky.get_emission(freq=bpasses['freq_ghz']*u.GHz , 
                                   weights= bpasses[det]  ) 
        alms = hp.map2alm(maps =signals ,lmax =lmax, mmax=mmax ) 
        ### to avoid any issue with  fwhm, i consider all sky alms convolved with 10arcmin <17.8 arcmin of the highest reso LB channel 
        alms = hp.smoothalm( alms=alms ,fwhm= pl.radians(10/60.)  ,mmax=mmax ) 
        almT = np.zeros_like(alms ) 
        almT[0] = alms[0] 
        almEB = np.zeros_like(alms ) 
        almEB [1:] = alms[1:] 
        almBE = np.zeros_like(alms ) 
        almBE [2] = -alms[1] 
        almBE [1] = alms[2]

        hp.write_alm(filename=almTfile  , alms =almT , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
        hp.write_alm(filename=almEBfile , alms =almEB , lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
        hp.write_alm(filename=almBEfile , alms =almBE, lmax=lmax, mmax=mmax , mmax_in=mmax, overwrite=True )
    
        end= time.perf_counter() 
        #print(end-start)
        #if idet >1: break 
    #break 
comm.Barrier()

comm.Disconnect 






