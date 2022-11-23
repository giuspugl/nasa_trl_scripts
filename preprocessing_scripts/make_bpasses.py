import toml 
import pylab as pl 
import litebird_sim as lbs 
import astropy 
from astropy import units as u 
import os 
import glob 
hwfiles= glob.glob("*.toml.gz")

bpassdir = "/Users/peppe/work/satellite_sims/sampled_bands/"

for hwfile in hwfiles: 
    dic_lb = toml.load(hwfile)
    
    bandstring = [ k for k in dic_lb['bands'].keys()  ][0]
    print(bandstring)
    fwhm = dic_lb['bands'][bandstring]['fwhm'] *u.arcmin 
    b0 = dic_lb['bands'][bandstring]['center'] *u.GHz
    bw = dic_lb['bands'][bandstring]['bandwidth'] *u.GHz 
    btype= "cheby"
    Band = lbs.bandpasses.BandPassInfo( bandcenter_ghz=band['center'], 
                                        bandwidth_ghz=band['bandwidth'],
                                        bandtype=btype,
                                        normalize=True,
                                       nsamples_inband=64)
    Band.bandpass_resampling (nresample=50,bstrap_size=5000  ) 
    sampled_band ={ det:Band.bandpass_resampling (nresample=50,bstrap_size=5000, model=Band.model    ) 
                   for det in dic_lb['detectors'] .keys()  }
    pl.savez(f"{bpassdir}/{bandstring}_{btype}.npz", **sampled_band , freq_ghz =Band.freqs_ghz )
            
    
 


