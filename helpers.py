import numpy as np
import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling,transform_bounds,transform_geom
from rasterio.transform import Affine
import restee as ree

def tile256(raster_path,output_dir):    
    """
    Create raster tiles of 256 by 256
    Arguments:
        raster_path: input raster path
        output_dir: output folder path   
    Returns:
        256 by 256 raster tiles    
    """
    #Make outdir if is not created
    os.makedirs(output_dir, exist_ok=True)
    
    
    with rasterio.open(raster_path) as src: 
        width = src.width 
        height = src.height 
        res =  src.transform[0]
        UL_c,UL_r = src.transform * (0, 0)   
        band = src.read(1)       
        kwargs = src.meta.copy()
                   
        for col in range(int(np.round_(width/256))):           
            for row in range(int(np.round_(height/256))):
                tf_tile = Affine(res, 0.0,UL_c+(256*res*col),0.0,-1*(res),UL_r+(256*(-1*(res))*row))   
                id = int(raster_path.split(os.sep)[-2].split('_')[-1])
                tile_name = f'{id:03d}_{row:01d}{col:01d}.tif'            
                np_arr = np.zeros((256, 256), dtype='float32')                
                tile_band = band[256*row:256*row+256,256*col:256*col+256]              
                np_arr[:tile_band.shape[0], :tile_band.shape[1]] = tile_band
                
                kwargs.update({
                    'transform': tf_tile,
                    'width': 256,
                    'height': 256,
                    'compress': 'lzw',
                    'dtype':'float32',})
                                
                with rasterio.open(os.path.join(output_dir,tile_name), "w", **kwargs) as dst:
                    dst.write(np_arr,1)

        np_arr = None 
        tile_band = None
        band = None
            
def restgee_data(input_tile,ee_img,ee_img_band,output_dir,restee_session):
    """
    Download GEE objectecs correspond to raster tile extent using EE REST API(restee package)
    Arguments:  
        input_tile: Input calibrated raster tile 
        ee_img: GEE object
        ee_img_band: GEE object band name
        output_dir: output directory path
        restee_session : rest ee session
    Returns:
        GEE objectecs as tiles correspoding to calibrated raster tiles
    """
    with rasterio.open(input_tile) as src:
        kwargs = src.meta.copy()
        tile_tranform = src.transform
        tile_bounds = src.bounds
        tile_crs = src.crs     
        gee_domain = ree.Domain((tile_bounds[0],
                                 tile_bounds[1],
                                 tile_bounds[2],
                                 tile_bounds[3]),
                                resolution= src.transform[0],
                                crs =str(tile_crs))
        
        band_utm = np.int32(ree.img_to_ndarray(restee_session, gee_domain, image=ee_img, bands=ee_img_band))
        gdal_mask = src.dataset_mask()
        gdal_mask = np.int32(np.where(gdal_mask == 0, gdal_mask, 1))
        band_utm = band_utm*gdal_mask
        
        dst_tranform = rasterio.transform.from_bounds(tile_bounds.left, 
                                                      tile_bounds.bottom,
                                                      tile_bounds.right, 
                                                      tile_bounds.top, 
                                                      band_utm.shape[1],
                                                      band_utm.shape[0])      
        out_file_path = os.path.join(output_dir,os.path.basename(input_tile))
        with rasterio.open(out_file_path, "w", **kwargs) as dst:
            reproject(
                source=band_utm,
                destination=rasterio.band(dst,1),
                src_transform=dst_tranform,
                src_crs=tile_crs,
                dst_transform=tile_tranform,
                dst_crs=tile_crs,
                resampling=Resampling.nearest)            
            