import argparse
import os
import glob
import concurrent.futures
from google.auth.transport.requests import AuthorizedSession
import ee
import restee as ree
from tqdm import tqdm
from helpers import restgee_data

def main(args):
          
    #DEM dir creation
    os.makedirs(os.path.join(args.out_dir,'dem'), exist_ok=True)
    
    ee.Authenticate(auth_mode='notebook') 
    session = AuthorizedSession(ee.data.get_persistent_credentials())
    
    # Creating a restee session
    class EESessionContainer(ree.EESession):
        def __init__(self, project, session):
            self._PROJECT = project
            self._SESSION = session

    # Create an EESesssion object with the correct permissions
    ee_session = EESessionContainer(args.cld_projid, session)

    # Authenticate EE with the session credentials
    ee.Initialize(ee_session.session.credentials, project=args.cld_projid)
    
    # SRTM dem data
    elevation = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    
    # One of 2 polarization
    chip_list = sorted(glob.glob(os.path.join(args.in_dir,'*.tif'), recursive = True))
    with tqdm(total=len(chip_list),position=0, leave=True, desc="GEE data request progress: SRTM DEM") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            # Start the load operations and mark each future with its chip        
            future_to_chip = {executor.submit(restgee_data,chip,elevation, 'elevation', os.path.join(args.out_dir,'dem'),ee_session): chip for chip in chip_list}          
            for future in concurrent.futures.as_completed(future_to_chip):
                chip = future_to_chip[future]
                pbar.update(n=1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Downloading corresponding SRTM DEM tile from GEE for a given raster chip')
    parser.add_argument(
        "--cld_projid",
        required=True,
        type=str,
        help="Cloud project id",
    )  
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Input tile directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output folder for corresponding dem tiles",
    )
    args = parser.parse_args()
    
    main(args)