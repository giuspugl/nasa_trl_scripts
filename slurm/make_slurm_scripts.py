import pylab as pl 
import toml 
import warnings
import glob 
import time 
import os 
warnings.filterwarnings("ignore")
import argparse 


hwp_rpm ={'LFT':46,"MFT":39, "HFT": 61 } 

def make_script (args): 
    sim_dir= args.sim_dir 
    hw_dir =f'{sim_dir}/hardware/' 
    alm_dir =f'{sim_dir}/sky_inputs_alm/' 
    blm_dir = f"{sim_dir}/expanded_beams/"
    temp_dir =f'{sim_dir}/sky_templates/'

    
    hwfiles= glob.glob(f'{hw_dir}/*toml.gz') 
    os.system(f"rm  {sim_dir}/scripts/*.sh" ) 
     
    for hwfile in hwfiles   : 
        
        dic_lb = toml.load(hwfile)
        telescope= list(dic_lb['telescopes'].keys())[0] 
        band= list(dic_lb ['bands'].keys() )[0]
        print(f"script for {band} " )
        ftmp  = open(f'./slurm/run.slrm','r')
        scriptfile = open(f'{sim_dir}/scripts/{band}.slrm', 'w')
        
        for line in ftmp.readlines():
            scriptfile.write(line)
        ftmp.close()
        scriptfile.close()
        with open( f'{sim_dir}/scripts/{band}.slrm', "a+") as txtfile:
            txtfile.write( f'hwprpm={hwp_rpm[telescope]}  \n' )
            txtfile.write( f'ntask=1080  \n' )
            txtfile.write( f'outdir="{sim_dir}/{band}"  \n' )
            txtfile.write( f'blmfile="{blm_dir}/{band}/'+'{detector}.fits" \n' )
            txtfile.write( f'templatefile="{temp_dir}/template_map_T_{band}_top-hat_bpass_K_CMB.fits" \n' )
            txtfile.write( f'skyfile="{alm_dir}/{band}/'+'sky_alm_{detector}.fits" \n' )
            txtfile.write( f'hardware="{hw_dir}/{band}.hdf5" \n' )
            txtfile.write( f'pyscript="{sim_dir}/scripts/nasa_trl_scripts/toast_scripts/toast_sim_lb.py" \n' )
            txtfile.write( f'config="{sim_dir}/scripts/nasa_trl_scripts/toast_scripts/config_log.toml" \n' )
            txtfile.write( f'hardware="{hw_dir}/{band}.hdf5" \n' )

            txtfile.write( 'LOG_OUT="${outdir}/'+f'run_{band}_'+'${SLURM_JOB_ID}"  \n' )
 
            command="""
echo Calling srun at $(date) \n \n 
srun -n $ntask -N $nnode   ${pyscript}   --out_dir   ${outdir}  \
--schedule ${LBSIM}/input_data/schedule_3yrs.ecsv \
--focalplane  ${hardwre}  \
--config ${config} \
--sim_satellite.hwp_rpm ${hwprpm} \
--beam_convolution.beam_file ${blmfile} --beam_convolution.sky_file ${skyfile}\
--scan_temp.file ${templatefile} \
--mapmaker.output_dir ${outdir} --mapmaker.convergence 1e-12 \
--calibrator.convergence 1e-12 \
--job_group_size 3 >> ${LOG_OUT} 2>&1 \n 
            """
    

            txtfile.write(command )
            txtfile.write( f'echo End slurm script at $(date) \n' ) 
        txtfile.close()
        with open(f'{sim_dir}/scripts/{telescope}_submit_all.sh', "a+") as shfile : 
            shfile.write(f'sbatch {sim_dir}/scripts/{band}.slrm \n ')
        shfile.close() 
    os.system(f"chgrp litebird  -R {sim_dir}/scripts" ) 
    os.system(f"chmod g+x  {sim_dir}/scripts/*_submit_all.sh" ) 



        
if __name__=="__main__":
        parser = argparse.ArgumentParser( description="prepare slurm script  " )
        parser.add_argument("--sim_dir" ,    help='path of the main simulation directory', required=True)
        args = parser.parse_args()
        make_script(args) 
        
        
    
    