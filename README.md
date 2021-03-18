# GHS-S2Net
Is a repository of the all pre-trained models for built-up areas extraction from Sentinel-2 (4Bands 10 m resolution data).
The models were obtained by training the GHS-S2Net CNN architecture using detailed learning sets per UTM grid zone 
The method is described in Corbane et al. 2020 :
Corbane, C., Syrris, V., Sabo, F., Politis, P., Melchiorri, M., Pesaresi, M., Soille, P., Kemper, T., 2020. Convolutional neural networks for global human settlements mapping from Sentinel-2 satellite imagery. Neural Computing & Applications.. 
doi:10.1007/s00521-020-05449-7

# Dependencies 

Models were trained using python 3.7.4 and the following libraries:

tensorflow-gpu==2.0.0
Keras==2.3.1
