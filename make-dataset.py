import argparse
import glob
import numpy as np
import os
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS
import shutil
import tarfile
from tqdm import tqdm 
from helpers import tile256 

def main(args):
    # Creating project folder structure"
    dir_list = ['data','models']
    for dirc in dir_list:
        directory_path = os.path.join(args.proj_dir,dirc)
        os.makedirs(directory_path, exist_ok=True)
    
    with tarfile.open(args.chips) as tar:
        print(f'Reading file {os.path.basename(args.chips)}')
        member_list = tar.getmembers()
        for i in tqdm(range(len(member_list)),position=0, leave=True,desc="S1 chips extraction"):
            if member_list[i].name.endswith('.tif'):
                tar.extract(member_list[i],os.path.join(args.proj_dir,'data'))
            
    with tarfile.open(args.labels) as tar:
        print(f'Reading file {os.path.basename(args.labels)}')
        member_list = tar.getmembers()
        for i in tqdm(range(len(member_list)),position=0, leave=True,desc="S1 water labels extraction"):
            if member_list[i].name.endswith('.tif'):
                tar.extract(member_list[i],os.path.join(args.proj_dir,'data'))
                
    # Getting S1 vv,vh and water label chip file paths
    s1_vv_img_path_list = sorted(glob.glob(os.path.join(args.proj_dir,'data','c2smsfloods_v1_source_s1','*/','*VV.tif'), recursive = True))
    s1_vh_img_path_list = sorted(glob.glob(os.path.join(args.proj_dir,'data','c2smsfloods_v1_source_s1','*/','*VH.tif'), recursive = True))
    s1_img_label_path_list = sorted(glob.glob(os.path.join(args.proj_dir,'data','c2smsfloods_v1_labels_s1_water','*/','*.tif'), recursive = True))
    
    # Covert to 256_256 tiles
    for i in tqdm(range(len(s1_vv_img_path_list)),position=0, leave=True,desc="Tiling VV chips"):
        tile256(s1_vv_img_path_list[i],os.path.join(args.proj_dir,'data','chips','VV'))   
    for i in tqdm(range(len(s1_vh_img_path_list)),position=0, leave=True,desc="Tiling VH chips"):
        tile256(s1_vh_img_path_list[i],os.path.join(args.proj_dir,'data','chips','VH'))     
    for i in tqdm(range(len(s1_img_label_path_list)),position=0, leave=True,desc="Tiling water labels"):
        tile256(s1_img_label_path_list[i],os.path.join(args.proj_dir,'data','labels'))   
    print('Removing original c2smsfloods data')
    for dirc in glob.glob(os.path.join(args.proj_dir,'data','c2s*')):
        shutil.rmtree(dirc)
    print('Process completed')               
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting Cloud to Street - Microsoft flood dataset (Sentinel-1) chip (256*256) information to a CSV file')
    parser.add_argument(
        "--proj_dir",
        default=os.getcwd(),
        type=str,
        help="Project directory location",
    )  
    parser.add_argument(
        "--chips",
        type=str,
        required=True,
        help="file path for S1 chip data (tar.gz)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="file path for S1 label data (tar.gz)",
    )
    args = parser.parse_args()
    
    main(args)