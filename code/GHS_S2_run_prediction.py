#Code implementation for tile-based predictions as described in:
#Corbane, C., Syrris, V., Sabo, F., Politis, P., Melchiorri, M., Pesaresi, M., Soille, P., Kemper, T., 2020. Convolutional neural networks for global human settlements mapping from Sentinel-2 satellite imagery. Neural Computing & Applications. doi:10.1007/s00521-020-05449-7

####################
#The Python script gets as input the 4-band Sentinel-2 image composite in 10 m spatial resolution. The order of the bands is 0: Band 2–Blue; 1: Band 3–Green; 2: Band 4–Red; 3: Band 8–NIR 
#This composite is available at:
#https://data.jrc.ec.europa.eu/dataset/0bd1dfab-e311-4046-8911-c54a8750df79
#Corbane, Christina; Politis, Panagiotis (2020):  GHS-composite-S2 R2020A - Sentinel-2 global pixel based image composite from L1C data for the period 2017-2018. European Commission, Joint Research Centre (JRC) [Dataset] doi:10.2905/0BD1DFAB-E311-4046-8911-C54A8750DF79 PID: http://data.europa.eu/89h/0bd1dfab-e311-4046-8911-c54a8750df79

#Please consider all relevant papers if you use this code

import os, sys
from osgeo import gdal, osr
from osgeo import gdalconst
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from sklearn.feature_extraction import image
from skimage.filters import threshold_multiotsu, threshold_otsu
import matplotlib.pyplot as plt
import math
import copy
import time
import gc
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #prevent TF from printing 

from keras import backend as K
import keras
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Input
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator

def ReadImage(infile):
    src = gdal.Open(infile)
    projection = src.GetProjection()
    geotransform = src.GetGeoTransform()
    proj = osr.SpatialReference(wkt=src.GetProjection())
    EPSG = proj.GetAttrValue('AUTHORITY',1)
    datatype = src.GetRasterBand(1).DataType
    datatype = gdal.GetDataTypeName(datatype)
    ulx, xres, xskew, uly, yskew, yres  = src.GetGeoTransform()
    lrx = ulx + (src.RasterXSize * xres)
    lry = uly + (src.RasterYSize * yres)
 
    cols = src.RasterXSize
    rows = src.RasterYSize
    
    if src.RasterCount==1:
        Im = src.GetRasterBand(1).ReadAsArray()
    else:
        Im = np.zeros((src.RasterCount,rows,cols), dtype=np.uint16)
        for q in range(src.RasterCount):
            #print(q, end=" ")
            Im[q,:,:] = src.GetRasterBand(q+1).ReadAsArray()
  
    return Im, src, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection

batch_size = 100000
rows, cols = 10000, 10000
window = 5

#Example for UTM grid zone 32T:
model = load_model('path_to_the_trained_model' + '.h5') #https://github.com/ec-jrc/GHS-S2Net/blob/main/Pretrained_models/MODEL_CNN_32T.h5
infile = # https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A/GHS_composite_S2_L1C_2017-2018_GLOBE_R2020A_UTM_10/V1-0/32T/S2_percentile_30_UTM_271-0000023296-0000000000.tif

#Read the 4-band Sentinel-2 image
S2, srcS2, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection = ReadImage(infile) 

Datamask = np.sum(S2.reshape(S2.shape[0], S2.shape[1]*S2.shape[2])>0, axis=0)==4
I2 = np.rollaxis(S2, 0, 3) 
                
I2 = np.float32(I2)
del S2
            
I2[I2>10000] = 10000
I2 = I2 / 10000.

#Prepare (expand) tiles for the model prediction
#sliding window 5x5
for q in range(I2.shape[2]):
    print(q, end=" ")
    tmp = copy.copy(I2[:,:,q])
    tmp = np.pad(tmp, (int(window/2),int(window/2)), 'reflect')
    tmp = image.extract_patches_2d(tmp, (window, window))
    if q == 0:
        T2 = np.expand_dims(tmp, axis=3)
    else:
        T2 = np.concatenate((T2, np.expand_dims(tmp, axis=3)), axis=3)
del tmp
    
gc.collect()

#Run the prediction
tt = time.time()
Response = model.predict(T2, batch_size=batch_size*8, verbose = 0)
prediction_time = str(time.time() - tt)
                   
del T2
gc.collect()
                    
Response = Response * 10000
Response = Response.astype(np.uint16)
Response[~Datamask,:] = 65535
            
Out = Response[:, 1].reshape(rows, cols)              

#save the results
driver = gdal.GetDriverByName('GTiff')
dst = driver.Create('path_out' + 'filename' +  '.tif' , cols, rows, 1, gdal.GDT_UInt16, ['COMPRESS=DEFLATE', 'TILED=YES'])
dst.SetGeoTransform(geotransform)
dst.SetProjection(projection)
dst.GetRasterBand(1).SetNoDataValue(65535)
dst.GetRasterBand(1).WriteArray(Out)
dst.FlushCache()
dst = None
                
del Datamask, Response

gc.collect()