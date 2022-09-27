#Functions to run to optimize different camera configurations. 


import numpy as np
import tifffile as tf




############################################## Object Optimization #############################################


def object_placement(camera, z0, model, center = True):
    """Returns the proper alpha phs based on if focus is in the center or front of object"""

    

    return alpha_ph


def natural_vignetting(camera, z0s,model, falloff='linear'):
    """This computes the natural vignetting which occurs for objects off the optical axis (reference is found in Applied Photographic optics sidney
    Fall off can be distance squared if object is small and order of 10 diameters away"""
    D = camera.aperture_f1 / camera.f1
    A = np.pi*(D/2)**2 #Area of aperture/lense
    voxel_area = camera.forward_parameters[0]**2
    vox_size = camera.forward_parameters[0]
    Z, Y, X = model.shape
    mask = np.zeros(model.shape)
    if Y%2 != 0:
        #odd meaning center pixel
        y_values = np.linspace(-vox_size*(Y-1)/2,vox_size*(Y-1)/2,Y)
    else:
        y_values = np.linspace(-vox_size*Y/2+vox_size/2,vox_size*Y/2-vox_size/2,Y)
    if X%2 !=0:
        x_values = np.linspace(-vox_size*(X-1)/2,vox_size*(X-1)/2,X)
    else:
        x_values = np.linspace(-vox_size*X/2+vox_size/2,vox_size*X/2-vox_size/2,X)

    xx, yy = np.meshgrid(x_values,y_values)
    optical_axis = np.array([0,0,1])
    for index, z in enumerate(z0s):
        norm_vec = np.sqrt(xx**2+yy**2+z**2)
        vectors = np.array([xx,yy,z])
        dot = np.matmul(vectors,optical_axis)
        cos4 = (dot/norm_vec)**4
        ##Formula
        if falloff == 'linear':
            mask[index,:,:] = cos4/z
        else:
            mask[index,:,:] = cos4/z**2

    return mask
    

###############################################  CAMERA OPTIMIZATION ############################################

def num_MLA(camera):
    """Calculates the number of MLA lens given sensor size and MLA diameter has been selected"""
    size  = np.floor(camera.sensor_size / camera.pixel_size_ts)
    return size.astype(np.intp)
    

def z1_calculator(camera, z0):
    """Assumes focal length and z0 have been set"""
    return (np.abs(z0-camera.f1)/(camera.f1*z0))**-1

def image_side_fnumber(camera, z0):
    """Called from microlens contraints
    returns the main_lens's image_side fnumber"""
    D = camera.f1 / camera.aperture_f1
    return z0/D

def microlens_constraints(camera, z0):
    """Moves the microlens focal length to accomadate microlens diameter constraint and MLA constraint"""
    image_fn = image_side_fnumber(camera, z0)
    camera.aperture_f2 = image_fn
    camera.f2 = camera.aperture_f2*camera.pixel_size_ts[0]
    return camera

def FOV(camera,z0):
    """Calculate Field of View"""
    FOV = z0/camera.z1*camera.sensor_size
    print('field-of-view is ', FOV, ' mm')

def camera_checker(camera,z0):
    FOV(camera, z0)
    resolution = z0/camera.z1*camera.pixel_size_ts
    if all(resolution < 0.3):
        print("Resolution of image is ", resolution, ' mm')
    else:
        print('resolution worse than EPID')
        print("Resolution of image is ", resolution, ' mm')
    if z0 < 4*camera.f1:
        print('object is probably too close to camera to be able to focus')
    print('camera z1 ', camera.z1)


def lytro_calibration(lf, mag):
    """Returns lf and z0"""
    lf.camera.aperture_f2 = 1.875  #MLA f#
    true_mag = 2*mag
    z0 = (lf.camera.f1 * (1+true_mag))/true_mag
    lf.camera.z1 = abs(1/lf.camera.f1-1/z0)**-1
    return lf, z0


def forward_model(MLA_diameter, MLA_num, focal_length, z0, model):
    """Will generate the forward projections"""
    pass




######################################################## IMAGE saving #############################################################

def metadata_creator(camera, z0, algorithm):
    data = {}
    data['focal length'] = camera.f1
    data['f#'] = camera.aperture_f1
    data['Resolution'] = camera.pixel_size_ts*z0/camera.z1
    data['Algo'] = algorithm
    return data

def image_saver(image, filename, meta):
    """Image is the lf_g object
    filename is string without tif
    meta is metadata dict object"""
    tf.imwrite('{}.tif'.format(filename), image.astype('float16'), )
    #if isinstance
    #tf.imwrite()