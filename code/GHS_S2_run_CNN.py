#Model architecture:
#Corbane, C., Syrris, V., Sabo, F., Politis, P., Melchiorri, M., Pesaresi, M., Soille, P., Kemper, T., 2020. Convolutional neural networks for global human settlements mapping from Sentinel-2 satellite imagery. Neural Computing & Applications.. doi:10.1007/s00521-020-05449-7
#Syrris, V.; Hasenohr, P.; Delipetrev, B.; Kotsev, A.; Kempeneers, P.; Soille, P. Evaluation of the Potential of Convolutional Neural Networks and Random Forests for Multi-Class Segmentation of Sentinel-2 Imagery. Remote Sens. 2019, 11, 907. https://doi.org/10.3390/rs11080907
####################
#Code uses the Sentinel-2 image composite of 10 m resolution and with 4 bands:
#https://data.jrc.ec.europa.eu/dataset/0bd1dfab-e311-4046-8911-c54a8750df79
#Corbane, Christina; Politis, Panagiotis (2020):  GHS-composite-S2 R2020A - Sentinel-2 global pixel based image composite from L1C data for the period 2017-2018. European Commission, Joint Research Centre (JRC) [Dataset] doi:10.2905/0BD1DFAB-E311-4046-8911-C54A8750DF79 PID: http://data.europa.eu/89h/0bd1dfab-e311-4046-8911-c54a8750df79

#Please consider all relevant papers if you use this code

import os, sys
from osgeo import gdal, osr
from osgeo import gdalconst
import numpy as np
import math
import copy
import time
import json
import pandas as pd
import gc

import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, core
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Read the imagery
#Input argument is the path to the raster/multiband image
#Output arguments are the multidimensional numpy array, image extent (ulx, uly, lrx, lry), pixel size (xres, yres), EPSG, geotransform and projection info 
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
        Im = np.zeros((src.RasterCount,rows,cols))
        for q in range(src.RasterCount):
            Im[q,:,:] = src.GetRasterBand(q+1).ReadAsArray()
    return Im, src, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection
	
#Round the numpy array to multiple of window size	
def myround(x, base=5):
    return base * round(x/base)

#Reshape the array into patches
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))	

#Model topology
#Input argument is the input array shape
#Output argument is the compiled model
def getModel(input_shape):
        
    model = Sequential()
    
    # CNN 1
    model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1), input_shape=input_shape))
    model.add(Conv2D(128, kernel_size=(2, 2), strides=(1, 1))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))

    # CNN 2
    model.add(Conv2D(512, kernel_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(512, kernel_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))

    model.add(Flatten())

    #Dense 1
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.1))

    # Output 
    model.add(Dense(2, activation="sigmoid"))
    optimizer = Adam(lr=0.0001, decay=0.0)
   
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

#Warp the reference rasters and corresponding datamasks to the Sentinel-2 extent
#Input arguments are the paths to the Sentinel-2 image and the paths to the corresponding reference data and datamasks
#Output arguments are the Sentinel-2 image and the Reference data (arrays), warped to the same extent
def warpImages(infile, inRaster_esm, inRaster_mt, inRaster_fb, inRaster_msoft, inRaster_fb_dm, inRaster_msoft_dm):
    S2, srcS2, ulx, uly, lrx, lry, xres, yres, EPSG, geotransform, projection = ReadImage(infile)
    
    Ref_esm = gdal.Warp('', inRaster_esm, format = 'MEM', outputType=gdal.GDT_Byte, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])
    Ref_esm = Ref_esm.ReadAsArray()
    Ref_esm = np.uint8(Ref_esm)
    esm_dm =  np.uint8(Ref_esm)
    esm_dm = Ref_esm > 0
    Ref_esm [Ref_esm < 255] = 0
    Ref_esm [Ref_esm == 255] = 1    

    Ref_fb = gdal.Warp('', inRaster_fb, format = 'MEM', outputType=gdal.GDT_Byte, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])

    Ref_fb = Ref_fb.ReadAsArray()
    Ref_fb = np.uint8(Ref_fb)

    fb_dm = gdal.Warp('', inRaster_fb_dm, format = 'MEM', outputType=gdal.GDT_Byte, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])
    fb_dm = fb_dm.ReadAsArray()
    fb_dm = np.uint8(fb_dm)

    Ref_mt = gdal.Warp('', inRaster_mt, format = 'MEM', outputType=gdal.GDT_Byte, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])
    Ref_mt = Ref_mt.ReadAsArray()
    Ref_mt = np.uint8(Ref_mt)
    mt_dm = Ref_mt > 0
    Ref_mt[Ref_mt<3] = 0
    Ref_mt[Ref_mt>2] = 1

    Ref_msoft = gdal.Warp('', inRaster_msoft, format = 'MEM', outputType=gdal.GDT_Byte, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])
    Ref_msoft = Ref_msoft.ReadAsArray()
    Ref_msoft = np.uint8(Ref_msoft)

    msoft_dm = gdal.Warp('', inRaster_msoft_dm, format = 'MEM', outputType=gdal.GDT_UInt16, dstSRS='EPSG:'+EPSG, xRes=xres, yRes=yres, outputBounds=[ulx, lry, lrx, uly])
    msoft_dm = msoft_dm.ReadAsArray()
    msoft_dm = np.uint8(msoft_dm)
    msoft_dm [msoft_dm == 1] = 0
    msoft_dm [msoft_dm == 40] = 1 
    msoft_dm [msoft_dm == 238] = 1 
    msoft_dm [msoft_dm == 233] = 1 
    msoft_dm [msoft_dm == 234] = 1
    msoft_dm [msoft_dm > 1] = 0

    mt_dm [esm_dm == 1] = 0
    mt_dm [msoft_dm == 1] = 0
    mt_dm [fb_dm == 1] = 0

    fb_dm [msoft_dm == 1] = 0
    fb_dm [esm_dm == 1] = 0

    Ref_fb [fb_dm == 0] = 0
    Ref_mt [mt_dm == 0] = 0
    Ref_msoft [msoft_dm == 0] = 0

    Ref = Ref_fb | Ref_mt
    Ref = Ref | Ref_msoft
    Ref = Ref | Ref_esm
    
    del Ref_fb, Ref_mt, Ref_esm, Ref_msoft, mt_dm, msoft_dm, fb_dm, esm_dm
    
    return S2, Ref

#Prepare the Sentinel-2 (X) and Reference data (labels) (Y) for the model input shape dimensions
#Input arguments are the Sentinel-2 image (array), Reference data (array), path to the logfile and the window size
#Output arguments are the reshaped arrays (X, Y). X_bu is an optional output which defines the number of built-up patches in the UTM zone (if there are no labels in the UTM zone, the training will be skipped)
def prepareXY(S2, Ref, logfile, window_size):
    
    rows = S2.shape[1]
    cols = S2.shape[2]

    if ((cols % window_size) > 0 or (rows % window_size))> 0:
        cols = myround(cols-3, window_size)
        rows = myround(rows-3, window_size)
        Ref = Ref[0:rows,0:cols]
        S2 = S2[:,0:rows,0:cols]

    Y = Ref
    del Ref

    logfile.write ("S2 shape: %s, %s, %s\n" % (S2.shape[0], S2.shape[1], S2.shape[2]))

    Datamask = np.sum(S2.reshape(S2.shape[0], S2.shape[1]*S2.shape[2])>0, axis=0)==4

    Y = Y.flatten()
    Y[~Datamask] = 0
    Y = Y.reshape(rows, cols)

    Train_Y = blockshaped(Y, window_size, window_size)

    X = np.zeros((S2.shape[0], Train_Y.shape[0], Train_Y.shape[1], Train_Y.shape[2]))
    for q in range(S2.shape[0]):
        X[q, :, :, :] = blockshaped(S2[q,:,:], window_size, window_size)
    X.shape

    del S2

    X = np.float32(X)
    X[X>10000] = 10000
    X = X / 10000.
    X.shape

    X = np.rollaxis(X, 0, 4)
    xshape = X.shape

    index_no_data = np.argwhere(np.any(X[:,:,:,0].reshape(X.shape[0], X.shape[1]*X.shape[1]), axis = 1) == 0)
    logfile.write("no data count: %s\n" % index_no_data.shape[0])
    datamask = np.ones(X.shape[0],dtype=bool)
    datamask[index_no_data] = False

    Train_Y = Train_Y[datamask]                

    idx_no_bu = np.argwhere(np.sum(Train_Y.reshape(Train_Y.shape[0], Train_Y.shape[1]*Train_Y.shape[2]), axis = 1) == 0)
    idx_bu = np.argwhere(np.sum(Train_Y.reshape(Train_Y.shape[0], Train_Y.shape[1]*Train_Y.shape[2]), axis = 1) > 0)

    permutation = np.random.permutation(idx_no_bu.shape[0])

    Y = np.zeros(Train_Y.shape +  (len(classes),), dtype=np.uint8)
    for q in range(len(classes)):
        tmp = Y[:,:,:,q]
        tmp[Train_Y==q] = 1
        Y[:,:,:,q] = tmp


    Train_Y = Train_Y[:, math.floor(window_size/2), math.floor(window_size/2)]
    Y = keras.utils.to_categorical(Train_Y, len(classes))

    X = X[datamask]

    idx_no_bu = idx_no_bu[permutation]
    idx_no_bu = idx_no_bu[0:int(idx_no_bu.shape[0]*0.6)] #take 60% of non-bu cells

    X_nbu = X[idx_no_bu]
    Y_nbu = Y[idx_no_bu]
    X_nbu = X_nbu[:,0,:,:,:]
    Y_nbu = Y_nbu[:,0,:]
    
    logfile.write("total cells: %s\n" % X.shape[0])
    logfile.write("non-built-up cells: %s\n" % X_nbu.shape[0])

    X_bu = X[idx_bu]
    Y_bu = Y[idx_bu]
    X_bu = X_bu[:,0,:,:,:]
    Y_bu = Y_bu[:,0,:]
    
    logfile.write("bu cells: %s\n" % X_bu.shape[0])
    X = np.concatenate([X_nbu, X_bu], axis = 0)
    Y = np.concatenate([Y_nbu, Y_bu], axis = 0)

    permutation = np.random.permutation(X.shape[0])
    X = X[permutation]
    Y = Y[permutation]
    
    return X,Y, X_bu

input_shape = (5, 5, 4)
batch_size = 200000 
epochs = 25

earlyStopping = EarlyStopping(monitor='val_loss', patience=6, verbose=0, mode='min')
mcp_save = ModelCheckpoint(fnmodel, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

################################################################################################################################################################
       
inRaster_esm = #path to the ESM
inRaster_mt = #path to the Landsat MT 
inRaster_fb = #path to the Facebook HRSL
inRaster_msoft =  #path to the Microsoft BU

n = 2
#datamasks
inRaster_msoft_dm = #path to the Microsoft datamask
inRaster_fb_dm = #path to the Facebook datamask

UTM_zones = #list of UTM zones to be processed (.txt)
UTM_zones_water = #List of UTM 'water' zones (.txt)

window_size = 5

X_stacked = np.zeros((1, window_size, window_size, 4))
Y_stacked = np.zeros((1, 2))
printcounter = 0

#Run the main workflow
with open (UTM_zones) as zones:
    folders = zones.read().splitlines()
    
with open (UTM_zones_water) as zones:
    folders_water = zones.read().splitlines()

for folder in folders:
    if folder in folders_water:
        n = 1
    else:
        n = 2
    tt = time.time() 
    utm_zone = folder.split('/') [7]
    logfile = open('path_to_the_log_file' + utm_zone +'.log', 'w')
    logfile.write('n is %s\n' % n)
    for subdir, dirs, files in os.walk(folder):
        for i,file in enumerate(files):
            if i % n == 0:
                filepath = subdir + os.sep + file
                
                if (printcounter == 50):
                    print('Still processing: ' + filepath)
                    printcounter = 0
                printcounter += 1
                
                if filepath.endswith(".tif") & bool("S2_percentile" in file):
                    infile = filepath
                    logfile.write("%s\n" % filepath)
                    
                    try:
                        S2, Ref = warpImages(infile, inRaster_esm, inRaster_mt, inRaster_fb, inRaster_msoft, inRaster_fb_dm, inRaster_msoft_dm)
                    
                        classes = np.unique(Ref)

                        logfile.write("classes: %s\n" % classes)
                    
                        if max(classes) < 1:
                            continue
                    
                        X, Y, X_bu = prepareXY(S2, Ref, logfile, window_size)
                        
                    except Exception as e: 
                        print ("Error caught in the data preparation: " + str(e) + " in " + filepath)
                        logfile.write("\nError: %s in %s" % (e, folder))
                        logfile.close()
                        continue
                    
                    if X_bu.shape [0] > 0:
                        X_stacked = np.concatenate([X_stacked, X], axis = 0) 
                        Y_stacked = np.concatenate([Y_stacked, Y], axis = 0)
                    else:
                        continue
                             
                    del X,Y
                    
    logfile.write("Y_Stacked shape: %s\n" % Y_stacked.shape[0])
    logfile.write("X_stacked shape: %s\n" % X_stacked.shape[0])
    logfile.write('\nElapsed time for data prep: %.02f sec' % (time.time() - tt))
    
    permutation = np.random.permutation(X_stacked.shape[0])
    X_stacked = X_stacked[permutation]
    Y_stacked = Y_stacked[permutation]
    
    del permutation
    del S2

    # Train the model on selected UTM grid zone 
    fnmodel = 'path_to_the_model_directory' + utm_zone +'.h5'
    mcp_save = ModelCheckpoint(fnmodel, save_best_only=True, monitor='val_loss', mode='min')
    
    if X_stacked.shape[0] > 5:
        try:
            model = getModel(input_shape) # load model and train from scratch
            
            t = time.time()
            history = model.fit(X_stacked, Y_stacked, batch_size=batch_size, epochs=epochs, verbose=0,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.1)

            hist_df = pd.DataFrame(history.history) 
            hist_json_file = 'path_to_the_model_directory' + utm_zone +'.json'
            with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f)
                
            logfile.write('\nElapsed time for training: %.02f sec' % (time.time() - t))
            
        except Exception as e: 
            print ("Error caught in the traning phase: " + str(e) + " in " + filepath)
            logfile.write("\nError: %s in %s" % (e, folder))
            logfile.close()
            continue
    else:
        continue
        
    X_stacked = np.zeros((1, 5, 5, 4))
    Y_stacked = np.zeros((1, 2))
    gc.collect()
    logfile.close()
    
