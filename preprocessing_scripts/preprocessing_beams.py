import astropy 
from numpy import pi  as pi
from astropy.io  import fits 
import numpy as np 
import pylab as pl 
import healpy as hp 
import os 
import glob 
import toast 
from toast import qarray as quat 
import litebirdtask  as lbt 
import time 

def edit_Hiroaki_sims(file ):
    ### Old sims of Hiroaki show an ICOMP parameter set to -3 we modify that to ICOMP=3 
    ### to make it compatible to grasp2stokes 
    
    
    with open(file, "r") as txtfile:
        c=0
        new_file_content = ""
        for  line in (txtfile):
            if c==9 : 
                strippedline = line.strip() 
                newline = strippedline.replace('-3', '3')
            else: 
                newline= line.strip() 

            new_file_content += newline +"\n"
            c+=1

    txtfile.close() 

    with open(file , "w") as outfile : 
        outfile.write(new_file_content) 
    outfile.close()
def transform_polar2cartesian(r , phi ) : 
    return r*np.cos(phi), r*np.sin(phi)
def transform_cartesian2polar (x , y ) : 
    return np.sqrt(x**2+y**2) ,  np.arctan(y/x)

def get_coordinates_from_LFT_sims(file): 
    xstring=file.split('/')[-1].split('_')[0] 
    ystring=file.split('/')[-1].split('_')[1]
    if xstring[0]=='m' and ystring[0]=='m':
        x= -1 *float(xstring[1:]); y= -1 *float(ystring[1:])
    elif xstring[0]=='m' and ystring[0]=='p':
        x= -1 *float(xstring[1:]); y= float(ystring[1:])
             
    elif xstring[0]=='p' and ystring[0]=='m':
        x=  float(xstring[1:]); y= -1 *float(ystring[1:])
             
    elif  xstring[0]=='p' and ystring[0]=='p':
        x=  float(xstring[1:]); y=  float(ystring[1:])
        
    
    r,phi = transform_cartesian2polar(x,y)   
    theta = np.radians(r *tele_plate_scale["LFT"] ) 
    return theta,phi 

def get_coordinates_from_MHFT_sims(file, telescope ): 
    pos=file.find("pix")+3
    idf = (file[ pos:pos+4 ]) 

    radiisims  = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160, 170, 175, 180])  #mm 
    idfile =  np.array(['{:04}'.format(i) for i in range(len(radiisims))] ) 
    idmask= idfile ==idf 
    thetaref =np.radians( radiisims[idmask][0]  *tele_plate_scale[telescope])
    return thetaref, 0. 

    
def make_stokes_beam_files (beamdir ) :
    
     
    for graspfile in glob.glob(f'{beamdir}/LFT/L*/*_tp.grd') :
        ### Hiroaki Old sims of Hiroaki show an ICOMP parameter set to -3 
        theta,phi = get_coordinates_from_LFT_sims(graspfile)
        pos = graspfile.find(".grd") 
        stokesfile = (graspfile[:pos]+f"_fpcoord_{theta:.5f}_{phi:.5f}.fits" ) 
        if "L3" in graspfile or "L4" in graspfile :
            continue 
            print(  "Editing header of beam maps  to make it compatible with GRASP2STOKES "  )
            edit_Hiroaki_sims(graspfile) 
            params=f""" 
            grasp_file =  {graspfile}
            grasp_format = grd_polar
            grasp_copol = x
            grasp_norm = four_pi
            stokes_file_polar = !{stokesfile}
            """
        # Hiroaki new sims for 32 mm pixels don't have this problem 
        elif "L2" in graspfile or "L1" in graspfile :
            params=f""" 
            grasp_file =  {graspfile}
            grasp_format = grd_polar 
            grasp_copol = x
            grasp_norm = four_pi
            stokes_file_polar = !{stokesfile}
            """
        text_file = open("grasp2stokes.par", "w")
        _ = text_file.write(params )
        text_file.close()
        os.system(f"{levels_exec}./grasp2stokes grasp2stokes.par >>log ")  

    for stringtele in ["MFT", "HFT"] : 
        
        for graspfile in glob.glob(f'{beamdir}/{stringtele}/*_tp.grd') : 
            theta,phi = get_coordinates_from_MHFT_sims(graspfile, stringtele)
            pos = graspfile.find(".grd") 
            stokesfile = (graspfile[:pos]+f"_fpcoord_{theta:.5f}_{phi:.5f}.fits" ) 

            params=f""" 
            grasp_file =  {graspfile}
            grasp_format = grd_polar
            grasp_copol = x
            grasp_norm = four_pi
            stokes_file_polar = !{stokesfile}
            """
            text_file = open("grasp2stokes.par", "w")
            _ = text_file.write(params )
            text_file.close()
            os.system(f"{levels_exec}./grasp2stokes grasp2stokes.par >>log ")  




def get_coords_graspsims(band ) :
    telescope= band[0]+"FT" 
    if telescope=="LFT":
        stokesfiles =glob.glob(f'{beamdir}/{telescope}/{band}/*.fits') 
        coordref= np.zeros((len(stokesfiles),2) ) 
        for i,f in enumerate(stokesfiles ): 
            startpos= f.find("coord_")
            endpos= f.find(".fits")
            coordstring = (f[startpos+6:endpos].split("_") )
            coordref[i ]  = np.float64(coordstring) 
            
    else:
        stokesfiles =glob.glob(f'{beamdir}/{telescope}/*.fits') 
        coordref= np.zeros((len(stokesfiles),2) ) 
        for i,f in enumerate(stokesfiles ): 
            startpos= f.find("coord_")
            endpos= f.find(".fits")
            coordstring = (f[startpos+6:endpos].split("_") )
            coordref[i ]  = np.float64(coordstring) 
     
    return coordref, stokesfiles  


def associate_grasp_beam (det, hw  ):
    detdic = hw.data['detectors'][det]
    pol_bolo  = detdic ['pol'] 
    detquat = detdic ['quat']
    band = detdic ['band']
    telesc= band[0] +"FT" 
    det_coord =quat.to_position(np.float64(   detquat )     ) 
    grasp_coord, grasp_files  = get_coords_graspsims(band) 
    id_grasp = np.argmin(np.linalg.norm(grasp_coord - det_coord , axis=1) )
    #print(f"GRASP:{grasp_coord [id_grasp] } DET:{det_coord}" ) 
    
    return grasp_files[id_grasp] ,   det_coord

def rotate_beam_map ( inputfile,   detcoordinates , detstring   ): 
    
    
    hdu=fits.open(f"{inputfile}")
    data=hdu[1].data
    IS = data.field(0).copy() 
    QS = data.field(1).copy() 
    US = data.field(2).copy() 
    VS = data.field(3).copy() 

    
    dphi = 2.*pi /nphi #GRASP beams are 2pi 
    
    if detstring[-1]== 'T':
        changesign=-1 
    else:
        changesign=1 
    #  GRASP Beams are on Pxx frame and they should refer to the bottom detector,
    # we change the sign of BeamdataQ  for the case of Top detector  

    
    if detstring[-1]!="L": 
        # estimate azimuthal offset of the detector for MHFT 
        # and rotate the beam map accordingly
        det_offset_theta,det_offset_phi  = detcoordinates[0], detcoordinates[1] 
        
        roll_offset = np.int_(np.ceil(det_offset_phi/dphi ) )
        rotate="roll" 
        
    else: 
        ### LFT beams are sampled on half focalplane of LFT so the beam maps in the other half 
        ### are simply obtained by flipping the x-axis of the input map.
        R = np.degrees(detcoordinates[0]) / platescale 
        cartcoords = np.array(transform_polar2cartesian(R,detcoordinates[0] ) ) 
        flipbeam =True if cartcoords[0] <0 else False 
        rotate="flip" 
        print(rotate, flipbeam) 
    
    listcols = [] 
     
    for istring, beam,factor  in zip (['', 'Q','U'], [IS, QS, US],[1,changesign,1] ) :
        beam= beam.reshape((ntheta, nphi) )
        if rotate=="roll" :      
            rolledbeam = np.roll (beam, roll_offset , axis=1)
        elif rotate=="flip" and flipbeam :
            rolledbeam = np.fliplr (beam)
        else :
            rolledbeam = beam 
        
        assert IS.shape== rolledbeam.ravel () .shape 
        
        listcols .append( fits.Column(name='Beamdata'+istring ,format='1E',array=rolledbeam.ravel()*factor ))
    
    listcols.append(fits.Column(name='BeamdataV',format='1E',array=VS) )
    coldefs=fits.ColDefs(listcols   )
    hdu1=fits.BinTableHDU.from_columns(coldefs)
    hdu1.header=hdu[1].header
    tmpname=f'/tmp/{detstring}.fits'
    hdu1.writeto(tmpname,overwrite=True)
    hdu.close() 
    
    return listcols 

def extract_side_lobes(dBcut = -50 , apodize=True ) :
    """ def get_sidelobe(  graspfile,detstring,  output_dir="./" ,  theta_cut=10 ,
                 theta_apo=5 , grid_size =(1001,1000 ), apodize=True   ): 
    hdu=fits.open(graspfile)
    data=hdu[1].data
    IS = data.field(0)
    QS = data.field(1)
    US = data.field(2)
    VS = data.field(3)
    ntheta = grid_size[0]
    nphi =grid_size[1]
    dtheta = np.pi /ntheta 
    ithetacut =np.int_ (np.radians(theta_cut)/dtheta)
    ithetapo =np.int_(ithetacut/2 ) 
    
    ## define the cut window function 
    
    W = np.zeros((ntheta,nphi))
    W[ :ithetacut,: ]=1 
    if apodize : 
        for i in range(ithetapo, ithetacut ): 
            W[i,:] =1./2. *(1+np.cos((i - ithetapo)/(ithetacut -ithetapo)*np.pi ))
    listcols = [] 
    
    for istring, beam  in zip (['', 'Q','U','V'], [IS, QS, US, VS ] ) :
        beam= beam.reshape((ntheta, nphi) ) *(1-W )
        assert IS.shape==  beam.ravel () .shape 
        listcols .append( fits.Column(name='Beamdata'+istring ,format='1E',array= beam.ravel()  ))
        
    coldefs=fits.ColDefs(listcols )
    hdu1=fits.BinTableHDU.from_columns(coldefs)
    hdu1.header=hdu[1].header
    name= f"{output_dir}/sl{theta_cut}deg_{detstring}.fits"
    hdu1.writeto(name,overwrite=True)
    hdu.close () 
    pass """
    return 

def run_beam2alm (  detstring , normalize=True, split_TP= False  ): 
     
    #run levelS.beam2alm 
    params= f"""  
    beam_main_file_polar = /tmp/{detstring}.fits
    beam_nphi={nphi}   
    beam_ntheta={ntheta}   
    beam_lmax={lmax}  
    beam_mmax={mmax }
    beam_alm_file= !/tmp/blm_{detstring}.fits 
    
    """
    text_file = open("beam2alms.par", "w")
    _ = text_file.write(params )
    text_file.close()
    os.system(f"{levels_exec}/./beam2alm beam2alms.par >> log ") 
    
    blm , mm = hp.read_alm(f"/tmp/blm_{detstring}.fits",return_mmax=True, hdu=[1,2,3]  ) 
    
    assert mm==mmax 
    
 
    if normalize : 
        #adopt the normalization convention of libconviqt 
        idx = hp.Alm.getidx (l=0, m=0 , lmax=lmax ) 
        
        norm = 2*pi *blm[0, idx] 
        blm/= norm 
        
    
    if split_TP : 
        blmT = np.array([ blm[0] , np.zeros_like(blm[0]),np.zeros_like(blm[0]) ] ).reshape(blm.shape) 
        blmP = np.array([np.zeros_like(blm[0]), blm[1], blm[2] ] ).reshape(blm.shape) 
        assert blmT.shape==blm.shape and blmP .shape==blm.shape 
        return blmT , blmP  
    
    else : 
        return blm 


 
    
    
    


tele_plate_scale={"LFT": 0.0476, 
        "MFT": 0.0875, 
        "HFT": 0.135 } #deg/mm 


lmax=1024 ; mmax=20
sim_dir="/global/cfs/cdirs/litebird/simulations/TOAST_nasa_trl_sims"
hwdir =f'{sim_dir}/hardware/' 
hwfiles= glob.glob(f'{hwdir}/*toml.gz') 

levels_exec="/global/u1/g/giuspugl/software/planck-levelS/linux_gcc/bin/"
nphi = 1000 
ntheta= 1001
beamdir = f"{sim_dir}/GRASP/"

#make_stokes_beam_files(beamdir) 
hwfiles= glob.glob(f'{hwdir}/*toml.gz') 
for hwfile in hwfiles : 
    hardware= lbt.Hardware ( hwfile )
    lb_bands = [b for b in hardware.data['bands'].keys()   ] [0]
    for det in hardware.data['detectors'].keys() : 
            start = time.perf_counter () 
            band =hardware.data['detectors'][det]['band'] 
            file,   detcoords =  associate_grasp_beam(det, hardware   ) 
            rotate_beam_map (detstring=det, inputfile=file , detcoordinates= detcoords) 
            blmT, blmP  = run_beam2alm(  detstring=det, normalize =True ,  split_TP=True ) 

            os.makedirs(f"/expanded_beams/{band}",exist_ok=True )
            hp.write_alm(filename=f"{sim_dir}/expanded_beams/{band}/{det}_T.fits",alms= blmT, lmax= lmax  , mmax = mmax , mmax_in=mmax , overwrite=True  )
            hp.write_alm(filename=f"{sim_dir}/expanded_beams/{band}/{det}_P.fits",alms= blmP   ,lmax= lmax  , mmax = mmax , mmax_in=mmax , overwrite=True )
            os.system("rm /tmp/*.fits" )
            end= time.perf_counter() 
            print(end-start ) 
            
    break



    
    


    
