import random
import os
import nibabel as nib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config
import tensorflow as tf
from scipy import ndimage
import skimage
from stl import mesh
from keras.models import load_model
from tensorflow.keras import backend as K
import pyvista as pv
import time
import vtk
from collections import OrderedDict
import nrrd


def get_data_and_points(data_folder, name):
    """ 
    This Function returns NII file as numpy array
    
    input:
    data_folder: folder in which NII file is stored
    name: name of NII file
    """
    img_path = os.path.join(data_folder, name)
    img_path_img = os.path.join(data_folder, [i for i in os.listdir(data_folder) if (i[-4:] == '.nii' in i)][0])
    epi_img = nib.load(img_path_img)
    epi_img_data = epi_img.get_fdata()
    
    return epi_img_data


def get_channels(arr, pos, num_channels):
    """
    This function returns the slice along with channels for the slice prediction model

    arr: numpy array
    pos: index of slice to be taken
    num_channels: number of channels to take on both side of the slice
    """
    return np.concatenate((arr[:,:,pos-num_channels:pos], np.expand_dims(arr[:,:,pos], 2), arr[:,:,pos+1:pos+1+num_channels]), axis=-1)



def center_and_crop_img(img, crop):
  """
  This function centers the centroid and then crops
  2D image.
  img: 2D slice
  crop: crop
  """
  # calculate the shift. Fist get the centroid of mass and then
  # subtract the centroid from the center to create shift coordinates.
  shift = np.array(img.shape)//2 - np.array(ndimage.center_of_mass(img+1024)).astype(int)

  # creating the crop points
  x1,x2 = img.shape[0]//2, crop[0]//2
  y1,y2 = img.shape[1]//2, crop[1]//2
  return ndimage.shift(img, shift = shift, order = 0, mode = 'nearest')[x1-x2:x1+x2,y1-y2:y1+y2]


'''
def data_generator_slice_prediction_v1_(epi_img_data, channels, crop, point):
    """
    This function will return batch of slices along with given number of channels and probabilites of the slices.
    data_folder: path to data folder
    channels:    number of channels to take along with the slice. Eg. If the value is 5 then it will take 5 channels
                 above and 5 channels below the given slice
    crop:        crop size. In this we set centroid of the slice as center and then cropping is done. Eg. (256,256).
    point:       Region around which slices needed to be taken. Valid arguments: "HipCenter" or "KneeCenter"
    """
    while True:
        if point == 'HipCenter':
            batch_size = len(list(range(channels, epi_img_data.shape[2]//3)))
            slices_to_take = list(range(channels, epi_img_data.shape[2]//3))
        else:
            batch_size = len(list(range(epi_img_data.shape[2]//3, (epi_img_data.shape[2]*2)//3)))
            slices_to_take = list(range(epi_img_data.shape[2]//3, (epi_img_data.shape[2]*2)//3))

        batch_x = np.zeros((batch_size, crop[0], crop[1], 2*channels+1, 1))
        batch_y_name = np.zeros((batch_size, 1))
        

        for index in range(batch_size):
            temp = get_channels(epi_img_data, slices_to_take[index], channels)

            for i in range(channels):
                batch_x[index,:,:,i,0] = center_and_crop_img(temp[:,:,i], crop)

            batch_y_name[index,0] = slices_to_take[index]

        return (batch_x/512.0, batch_y_name)
    
'''

def CROP_3D_CT_IMAGE_AT_GIVEN_POINT_NEW_256(img, centered_at, crop_size):
    crop_size = np.asanyarray(crop_size)
    img = np.pad(img, ((128,128),(128,128),(128,128)),'edge')   

    sp_idx = np.asarray(128+centered_at-crop_size//2,dtype=int)
    ep_idx = np.asarray(128+centered_at+crop_size//2,dtype=int)
    
    
       
    return img[sp_idx[0]:ep_idx[0], sp_idx[1]:ep_idx[1], sp_idx[2]:ep_idx[2]]

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def get_hip_knee_points_3d_cnn_model(epi_img_data):
    # z_size = epi_img_data.shape[2]
    # cm = ndimage.center_of_mass(epi_img_data[:,:,z_size//3:z_size*2//3]>200)    
    # cm = np.asarray(cm,dtype=int)  
    # cm[2] = z_size//3 + cm[2]
    # crop_center_knee = cm.copy()


    new_epi_img_data = epi_img_data.copy()
    new_epi_img_data[new_epi_img_data < 200] = 0
    new_epi_img_data[new_epi_img_data >= 200] = 1

    pnts = ndimage.center_of_mass(new_epi_img_data[:,:,epi_img_data.shape[2]//3:epi_img_data.shape[2]*2//3])
    pnts = np.array(pnts).astype(int)
    pnts[2] += epi_img_data.shape[2]//3

    knee_points = pnts.copy()

    # ankle points
    new_epi_img_data = epi_img_data.copy()
    new_epi_img_data = new_epi_img_data[:,:,::-1]
    cm = ndimage.center_of_mass(new_epi_img_data[:,:,new_epi_img_data.shape[2]*2//3:]>200)
    cm = np.asarray(cm,dtype=int)  
    cm[2] += new_epi_img_data.shape[2]*2//3

    img = np.pad(new_epi_img_data, ((128,128),(128,128),(128,128)),'edge')   
    crop_size_ = np.array([256,256,256])

    sp_idx = np.asarray(128+cm-crop_size_//2,dtype=int)
    ep_idx = np.asarray(128+cm+crop_size_//2,dtype=int)

    img = img[sp_idx[0]:ep_idx[0], sp_idx[1]:ep_idx[1], sp_idx[2]:ep_idx[2]]

    pnts_arr = []

    '''
    ankle_landmark_models_path = [r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\TIBIA_ANKLE_PREDICTION_3DCNN_MODEL_V0_0.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\TIBIA_ANKLE_PREDICTION_3DCNN_MODEL_V0_1.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\TIBIA_ANKLE_PREDICTION_3DCNN_MODEL_V0_2.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\TIBIA_ANKLE_PREDICTION_3DCNN_MODEL_V0_3.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\TIBIA_ANKLE_PREDICTION_3DCNN_MODEL_V0_4.h5']
    '''

    ankle_landmark_models_path = config.cnn_3d_point_prediction_models['Ankle']

    ankle_landmark_models = []
    for p in ankle_landmark_models_path:
        ankle_landmark_models.append(load_model(p, compile=False))
        ankle_landmark_models[-1].compile(optimizer = 'adam', loss=euclidean_distance_loss,run_eagerly=True)

    for model in ankle_landmark_models:
        pnts = model.predict(np.expand_dims(img[::2,::2,::2]/512.0,0))
        pnts = np.asarray(pnts,dtype=int)
        pnts_arr.append(pnts)

    pnts = np.array(pnts_arr).mean(0).astype(int)
    
    new_points = cm + pnts[0]

    new_points[2] = epi_img_data.shape[2] - new_points[2]
    ankle_pnts = new_points.copy()


    # hip point
    '''
    hip_landmark_models_path = [r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\HIP_CENTER_PREDICTION_3DCNN_MODEL_V0_0.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\HIP_CENTER_PREDICTION_3DCNN_MODEL_V0_1.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\HIP_CENTER_PREDICTION_3DCNN_MODEL_V0_2.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\HIP_CENTER_PREDICTION_3DCNN_MODEL_V0_3.h5',
                        r'D:\Bone_Landmark_Detection_And_Segmentation\notebooks\hip_ankle_models\HIP_CENTER_PREDICTION_3DCNN_MODEL_V0_4.h5']
    '''
    hip_landmark_models_path = config.cnn_3d_point_prediction_models['HipCenter']
    hip_landmark_models = []
    for p in hip_landmark_models_path:
        hip_landmark_models.append(load_model(p, compile=False))
        hip_landmark_models[-1].compile(optimizer = 'adam', loss=euclidean_distance_loss,run_eagerly=True)

    
    new_epi_img_data = epi_img_data.copy()
    new_epi_img_data = new_epi_img_data[:,:,::-1]
    cm = ndimage.center_of_mass(new_epi_img_data[:,:,:new_epi_img_data.shape[2]//3]>200)
    cm = np.asarray(cm,dtype=int)  

    img = np.pad(new_epi_img_data, ((128,128),(128,128),(128,128)),'edge')   
    crop_size_ = np.array([256,256,256])

    sp_idx = np.asarray(128+cm-crop_size_//2,dtype=int)
    ep_idx = np.asarray(128+cm+crop_size_//2,dtype=int)

    img = img[sp_idx[0]:ep_idx[0], sp_idx[1]:ep_idx[1], sp_idx[2]:ep_idx[2]]

    pnts_arr = []

    for model in hip_landmark_models:
        pnts = model.predict(np.expand_dims(img[::2,::2,::2]/512.0,0))
        pnts = np.asarray(pnts,dtype=int)
        pnts_arr.append(pnts)

    pnts = np.array(pnts_arr).mean(0).astype(int)
    
    new_points = cm + pnts[0]

    new_points[2] = epi_img_data.shape[2] - new_points[2]
    hip_pnts = new_points.copy()
    
    return hip_pnts, knee_points, ankle_pnts




def get_hip_knee_points(epi_img_data):
    """
    This function returns the Hip-Center points and Knee-Center points

    epi_img_data: numpy arr of NII file
    """
    x_upper, y_upper_name =data_generator_slice_prediction_v1_(epi_img_data, 5, (256,256), 'HipCenter')
    x_lower, y_lower_name = data_generator_slice_prediction_v1_(epi_img_data, 5, (256,256), 'KneeCenter')

    hip_slice_model = load_model(config.slice_prediction_models['HipCenter']) #remaining to change
    knee_slice_model = load_model(config.slice_prediction_models['KneeCenter'])

    y_pred_upper = hip_slice_model.predict(x_upper)
    y_pred_lower = knee_slice_model.predict(x_lower)

    temp = epi_img_data[:,:,int(y_lower_name[np.argmax(y_pred_upper)][0])]
    temp[temp<200] = 0
    temp_points = list(ndimage.center_of_mass(temp))
    temp_points.append(y_upper_name[np.argmax(y_pred_upper)][0])
    data_points_hip = np.array(temp_points).astype(int)

    temp = epi_img_data[:,:,int(y_lower_name[np.argmax(y_pred_lower)][0])]
    temp[temp<200] = 0

    temp_points = list(ndimage.center_of_mass(temp))
    temp_points.append(y_lower_name[np.argmax(y_pred_lower)][0])
    data_points_knee = np.array(temp_points).astype(int)

    return data_points_hip, data_points_knee


def load_segmentation_models():
    """
    This Function load the segmentation models and returns them as list
    """
    print('Loading Models for segmentation')

    def get_dice_loss(smoothing_factor):
        def dice_coefficient_loss(y_true, y_pred):
            y_true = y_true[:,:,:,:,1:]
            y_pred = y_pred[:,:,:,:,1:]
            Ncl = y_pred.shape[-1]
            w = np.zeros((Ncl,))
            for l in range(0,Ncl): w[l] = np.sum( np.asarray(y_true[:,:,:,:,l]==1,np.int8) )
            w = 1/(w**2+0.00001)
            
            flat_y_true = K.flatten(y_true)
            flat_y_pred = K.flatten(y_pred)
            return 1 - (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)
    
        return dice_coefficient_loss

    knee_models = []
    for path in config.knee_segmentation_models:
        model = load_model(path, compile=False)
        model.compile(optimizer = 'adam', loss=get_dice_loss(1),run_eagerly=True)
        knee_models.append(model)


    def get_dice_loss(smoothing_factor):
        def dice_coefficient_loss(y_true, y_pred):
            flat_y_true = K.flatten(y_true)
            flat_y_pred = K.flatten(y_pred)
            return 1 - (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

        return dice_coefficient_loss

    hip_models = []
    for path in config.hip_segmentation_models:
        model = load_model(path, compile=False)
        model.compile(optimizer = 'adam', loss=get_dice_loss(1),run_eagerly=True)
        hip_models.append(model)


    patella_models = []
    for path in config.patella_segmentation_models:
        model = load_model(path,compile=False)
        model.compile(optimizer='adam', loss=get_dice_loss(1), run_eagerly=True)
        patella_models.append(model)

    ankle_models = []
    for path in config.ankle_segmentation_models:
        model = load_model(path,compile=False)
        model.compile(optimizer='adam', loss=get_dice_loss(1), run_eagerly=True)
        ankle_models.append(model)

    return knee_models, hip_models, patella_models, ankle_models



def crop_around_given_point_without_padding_v1(data, data_points, crop_size):
    """
    This function Crops a 3d volumne of given crop_size around given data_points without using padding

    data: 3d numpy arr
    data_points: points around which crop is needed to be taken (eg. [256,126,49])
    crop_size: crop size (eg. [256,256,256])
    """
    crop = [[],[],[]]
    for i in range(3):
        if (data_points[i] - crop_size[i]//2) < 0:
            rem = abs(int(data_points[i] - crop_size[i]//2))
            crop[i].append(0)
            crop[i].append(int(data_points[i] + crop_size[i]//2) + rem )
        elif (data_points[i] + crop_size[i]//2) > data.shape[i]:
            rem = int(data_points[i] + crop_size[i]//2 - data.shape[i])
            crop[i].append(int(data_points[i] - crop_size[i]//2) - rem)
            crop[i].append(data.shape[i])
        else:
            crop[i].append(data_points[i] - crop_size[i]//2)
            crop[i].append(data_points[i] + crop_size[i]//2)

    return data[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], crop[2][0]:crop[2][1]], data, crop

def data_generator_seg_v1_hip_and_knee_(epi_img_data, crop_size, data_points_hip, data_points_knee, data_points_ankle):
        """
        This function generates data to predit the segmentation of hip and knee.
        This function will return data of half size of the given crop_size
        """
        batch_upper = np.zeros((1, crop_size[0], crop_size[1], crop_size[2],1))
        batch_lower = np.zeros((1, crop_size[0], crop_size[1], crop_size[2],1))
        batch_ankle = np.zeros((1, crop_size[0], crop_size[1], crop_size[2],1))

        hip_img_data, _, crop_upper = crop_around_given_point_without_padding_v1(epi_img_data, data_points_hip, crop_size)
        batch_upper[0,:,:,:,0] = hip_img_data

        knee_img_data, _, crop_lower = crop_around_given_point_without_padding_v1(epi_img_data, data_points_knee, crop_size)
        batch_lower[0,:,:,:,0] = knee_img_data

        ankle_img_data, _, crop_ankle = crop_around_given_point_without_padding_v1(epi_img_data, data_points_ankle, crop_size)
        batch_ankle[0,:,:,:,0] = ankle_img_data

        return batch_upper[:,::2,::2,::2,:]/512.0, batch_lower[:,::2,::2,::2,:]/512.0, batch_ankle[:,::2,::2,::2,:]/512.0, crop_upper, crop_lower, crop_ankle


def data_generator_seg_v1_patella_(epi_img_data, crop_size,data_points_knee):
    batch_lower = np.zeros((1, crop_size[0], crop_size[1], crop_size[2],1))
    knee_img_data, _, crop_lower = crop_around_given_point_without_padding_v1(epi_img_data, data_points_knee, crop_size)
    batch_lower[0,:,:,:,0] = knee_img_data

    return batch_lower[:,::2,::2,::2,:]/512.0, crop_lower


def get_bones_femur_tibia_fib(epi_img_data, knee_models, hip_models, patella_models, ankle_models, data_points_hip, data_points_knee, data_points_ankle):
    """
    This function creates binary masks of femur and tibia-fibula
    
    """
    upper_x, lower_x, ankle_x, crop_upper, crop_lower, crop_ankle = data_generator_seg_v1_hip_and_knee_(epi_img_data, (256,256,256), data_points_hip, data_points_knee, data_points_ankle)

    patella_x, patella_crop = data_generator_seg_v1_patella_(epi_img_data,(128,128,128), data_points_knee)

    pred_upper = None
    
    for model in hip_models:
        if pred_upper is not None:
            temp_pred =  model.predict(upper_x)
            pred_upper += temp_pred
        else:
            pred_upper = model.predict(upper_x)


    pred_lower = None
    for model in knee_models:
        if pred_lower is not None:
            temp_pred =  model.predict(lower_x)
            pred_lower += temp_pred
        else:
            pred_lower = model.predict(lower_x)


    pred_ankle = None
    for model in ankle_models:
        if pred_ankle is not None:
            temp_pred =  model.predict(ankle_x)
            pred_ankle += temp_pred
        else:
            pred_ankle = model.predict(ankle_x)


    pred_patella = None
    for model in patella_models:
        if pred_patella is not None:
            temp_pred =  model.predict(patella_x)
            pred_patella += temp_pred
        else:
            pred_patella = model.predict(patella_x)


    pred_patella /= len(patella_models)
    pred_upper /= len(hip_models)
    pred_lower /= len(knee_models)
    pred_ankle /= len(ankle_models)


    tibia_fib = pred_lower[0,:,:,:,-1]
    tibia_fib = ndimage.zoom(tibia_fib, zoom=2, order=1)
    tibia_fib[tibia_fib < 0.5] = 0
    tibia_fib[tibia_fib >= 0.5] = 1

    femur_knee = pred_lower[0,:,:,:,1]
    femur_knee = ndimage.zoom(femur_knee, zoom=2, order=1)
    femur_knee[femur_knee < 0.5] = 0
    femur_knee[femur_knee >= 0.5] = 1

    femur = pred_upper[0,:,:,:,1]
    femur = ndimage.zoom(femur, zoom=2, order=1)
    femur[femur < 0.5] = 0
    femur[femur >= 0.5] = 1

    patella = pred_patella[0,:,:,:,0]
    patella = ndimage.zoom(patella, zoom=2, order=1)
    patella[patella < 0.5] = 0
    patella[patella >= 0.5] = 1


    ankle = pred_ankle[0,:,:,:,0]
    ankle = ndimage.zoom(ankle, zoom=2, order=1)
    ankle[ankle < 0.5] = 0
    ankle[ankle >= 0.5] = 1


    ankle_new = np.zeros(ankle.shape)
    ankle_new[np.where(ankle == 0)] = 1

    
    tibia_fibula_data = np.zeros(epi_img_data.shape)
    tibia_fibula_data[crop_lower[0][0]:crop_lower[0][1], crop_lower[1][0]:crop_lower[1][1],crop_lower[2][0]:crop_lower[2][1]] = tibia_fib
    tibia_fibula_data[crop_ankle[0][0]:crop_ankle[0][1], crop_ankle[1][0]:crop_ankle[1][1],crop_ankle[2][0]:crop_ankle[2][1]] = ankle_new
    tibia_fibula_shaft = epi_img_data[:,:, crop_ankle[2][1]:crop_lower[2][0]]
    tibia_fibula_shaft[tibia_fibula_shaft < 200] = 0
    tibia_fibula_shaft[tibia_fibula_shaft >= 200] = 1
    tibia_fibula_data[:,:, crop_ankle[2][1]:crop_lower[2][0]] = tibia_fibula_shaft


    femur_data = np.zeros(epi_img_data.shape)
    femur_data[crop_lower[0][0]:crop_lower[0][1], crop_lower[1][0]:crop_lower[1][1],crop_lower[2][0]:crop_lower[2][1]] = femur_knee
    femur_data[crop_upper[0][0]:crop_upper[0][1], crop_upper[1][0]:crop_upper[1][1],crop_upper[2][0]:crop_upper[2][1]] = femur
    femur_shaft = epi_img_data[:,:, crop_lower[2][1]:crop_upper[2][0]]
    femur_shaft[femur_shaft < 200] = 0
    femur_shaft[femur_shaft >= 200] = 1
    # femur_shaft = ndimage.gaussian_filter(femur_shaft,10)
    # femur_shaft = np.round(femur_shaft)
    femur_data[:,:, crop_lower[2][1]:crop_upper[2][0]] = femur_shaft

    patella_data = np.ones(epi_img_data.shape)
    patella_data[patella_crop[0][0]:patella_crop[0][1], patella_crop[1][0]:patella_crop[1][1],patella_crop[2][0]:patella_crop[2][1]] = patella

    return femur_data, tibia_fibula_data, patella_data


def dilation_and_erosion(arr, dilation_iter=2, erosion_iter=1):
    arr_dilation = ndimage.binary_dilation(arr, iterations=dilation_iter)
    arr = ndimage.binary_erosion(arr_dilation, iterations=erosion_iter)
    arr = arr.astype(int)
    return arr_dilation.astype(int)


def save_nrrd(arr, name, nrrd_file_header):
    # pass
    nrrd.write(f'{name}.nrrd', arr, nrrd_file_header)


def getLargestCC(segmentation):
    """
    This function will remove all blobs other than the largest blob
    """
    labels = skimage.measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def Post_proccess_pred_seg_knee(p_mask, thr_bone_mask):
    
    im_size = thr_bone_mask.shape
    
    pelvis_mask = thr_bone_mask*(p_mask==0)
    pelvis_mask[:,:,:im_size[2]//2] = 0
    pelvis_mask = getLargestCC(pelvis_mask)
    
    foot_mask = thr_bone_mask*(p_mask==0)
    foot_mask[:,:,im_size[2]//2:] = 0
    foot_mask = getLargestCC(foot_mask)
    
    thr_bone_mask[pelvis_mask==1] = 0
    thr_bone_mask[foot_mask==1] = 0

    femur = thr_bone_mask*np.logical_not(np.logical_or(p_mask == 2 , p_mask==3))
    femur = getLargestCC(femur)
    tibia = thr_bone_mask*np.logical_not(np.logical_or(p_mask == 2 , p_mask==1))
    tibia = getLargestCC(tibia)
    patella = thr_bone_mask*np.logical_not(np.logical_or(p_mask == 1 , p_mask==3))
    patella = getLargestCC(patella)

    return femur, tibia, patella
    

def get_3D_mesh(arr, mesh_orig=None, save_name=None):
    """
    This function creates 3D mesh from array (.stl file is created)

    input:
    arr: 3D binary array
    save_name: save name
    """
    arr_ = None
    try:
        arr_ = getLargestCC(arr)
    except:
        pass
    
    if arr_ is not None:
        arr = arr_.copy()

    # verts, faces, _, _ = skimage.measure.marching_cubes(arr, 0.0)
    verts, faces, _, _ = skimage.measure.marching_cubes(arr, 0.0, spacing=(1, 1, 1))
    
    if mesh_orig is not None:
        verts = verts+mesh_orig
    arr = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            arr.vectors[i][j] = verts[f[j],:]

    temp_name = f'temp_mesh_{str(int(time.time()))}'
    arr.save(f'{temp_name}.stl')
    
    reader = pv.get_reader(f'{temp_name}.stl')
    mesh_ = reader.read()
    os.remove(f'{temp_name}.stl')

    smoothed_mesh = mesh_.smooth_taubin(n_iter=70)
    smoothed_mesh.save(f'{save_name}.stl')


def create_obj_file_from_stl(stl_files, save_name, colors):
    actors = []
    for i, name in enumerate(stl_files):
        reader = vtk.vtkSTLReader()
        #reader = vtk.vtkOBJReader()
        reader.SetFileName(name)

        mapper = vtk.vtkPolyDataMapper()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mapper.SetInput(reader.GetOutput())
        else:
            mapper.SetInputConnection(reader.GetOutputPort())


        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetPosition([0.0, 0.0, 0.0])
        actor.SetScale([1.0, 1.0, 1.0])

        #Changes the colour to purple for the first stl file 
        actor.GetProperty().SetColor(colors[i][0],colors[i][1],colors[i][2])

        actors.append(actor)

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    for actor_ in actors:
        renderer.AddActor(actor_)

    renderer.SetBackground(0, 0, 0)

    exporter = vtk.vtkOBJExporter()
    exporter.SetRenderWindow( renderWindow )
    exporter.SetFilePrefix( save_name ) #create mtl and obj file.
    exporter.Write()

def combine_stl_files(stl_files, combined_file, colors):
    combined_polydata = vtk.vtkPolyData()

    # Iterate over each STL file and merge into combined_polydata with color
    for i, name in enumerate(stl_files):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(name)
        reader.Update()

        # Apply color to the mesh
        color = colors[i]
        color_data = vtk.vtkUnsignedCharArray()
        color_data.SetNumberOfComponents(3)
        color_data.SetName("Colors")

        num_points = reader.GetOutput().GetNumberOfPoints()
        color_data.SetNumberOfTuples(num_points)
        for j in range(num_points):
            color_data.InsertTuple3(j, int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))

        polydata = reader.GetOutput()
        polydata.GetPointData().SetScalars(color_data)

        # Merge each STL file's polydata into combined_polydata
        appendFilter = vtk.vtkAppendPolyData()
        appendFilter.AddInputData(combined_polydata)
        appendFilter.AddInputData(polydata)
        appendFilter.Update()

        combined_polydata = appendFilter.GetOutput()

    # Write the combined polydata to a new STL file
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(combined_file)
    writer.SetInputData(combined_polydata)
    writer.Write()

def create_nrrd(org_header, arr, save_path):
    nrrd_file_header = OrderedDict()
    nrrd_file_header['type'] = 'unsigned char'
    nrrd_file_header['dimension'] = 3
    nrrd_file_header['space'] = 'left-posterior-superior'
    nrrd_file_header['sizes'] = org_header['sizes']
    nrrd_file_header['space directions'] = org_header['space directions']
    nrrd_file_header['kinds'] = ['domain', 'domain', 'domain']
    nrrd_file_header['encoding'] = 'gzip'
    nrrd_file_header['space origin'] = org_header['space origin']
    nrrd_file_header['Segment0_Color'] = '0.501961 0.682353 0.501961'
    nrrd_file_header['Segment0_ColorAutoGenerated'] = '1'
    # nrrd_file_header['Segment0_Extent'] = '260 446 150 360 22 918'
    nrrd_file_header['Segment0_ID'] = 'Segment_1'
    nrrd_file_header['Segment0_LabelValue'] = '1'
    nrrd_file_header['Segment0_Layer'] = '0'
    nrrd_file_header['Segment0_Name'] = 'F'
    nrrd_file_header['Segment0_NameAutoGenerated'] = '0'
    # nrrd_file_header['Segment0_Tags'] = 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'
    nrrd_file_header['Segment1_Color'] = '0.945098 0.839216 0.568627'
    nrrd_file_header['Segment1_ColorAutoGenerated'] = '1'
    # nrrd_file_header['Segment1_Extent'] = '260 446 150 360 22 918'
    nrrd_file_header['Segment1_ID'] = 'Segment_2'
    nrrd_file_header['Segment1_LabelValue'] = '2'
    nrrd_file_header['Segment1_Layer'] = '0'
    nrrd_file_header['Segment1_Name'] = 'T'
    nrrd_file_header['Segment1_NameAutoGenerated'] = '0'
    # nrrd_file_header['Segment1_Tags'] = 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'
    nrrd_file_header['Segment2_Color'] = '0.694118 0.478431 0.396078'
    nrrd_file_header['Segment2_ColorAutoGenerated'] = '1'
    # nrrd_file_header['Segment2_Extent'] = '260 446 150 360 22 918'
    nrrd_file_header['Segment2_ID'] = 'Segment_3'
    nrrd_file_header['Segment2_LabelValue'] = '3'
    nrrd_file_header['Segment2_Layer'] = '0'
    nrrd_file_header['Segment2_Name'] = 'P'
    nrrd_file_header['Segment2_NameAutoGenerated'] = '0'
    # nrrd_file_header['Segment2_Tags'] = 'Segmentation.Status:inprogress|TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SCT^85756007^Tissue~SCT^85756007^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'
    nrrd_file_header['Segmentation_ContainedRepresentationNames'] = 'Binary labelmap|Closed surface|'
    # nrrd_file_header['Segmentation_ConversionParameters'] = """Decimation factor|0.0|Desired reduction in the total number of polygons. Range: 0.0 (no decimation) to 1.0 (as much simplification as possible). Value of 0.8 typically reduces data set size by 80% without losing too much details.&Smoothing factor|0.5|Smoothing factor. Range: 0.0 (no smoothing) to 1.0 (strong smoothing).&Compute surface normals|1|Compute surface normals. 1 (default) = surface normals are computed. 0 = surface normals are not computed (slightly faster but produces less smooth surface display).&Joint smoothing|0|Perform joint smoothing.&Reference image geometry|-0.6855469999999999;0;0;137.30000299999998;0;-0.6855469999999999;0;175.5;0;0;0.9999999505376345;-170.699997;0;0;0;1;0;511;0;511;0;930;|Image geometry description string determining the geometry of the labelmap that is created in course of conversion. Can be copied from a volume, using the button.&Oversampling factor|1|Determines the oversampling of the reference image geometry. If it's a number, then all segments are oversampled with the same value (value of 1 means no oversampling). If it has the value "A", then automatic oversampling is calculated.&Crop to reference image geometry|0|Crop the model to the extent of reference geometry. 0 (default) = created labelmap will contain the entire model. 1 = created labelmap extent will be within reference image extent.&Collapse labelmaps|1|Merge the labelmaps into as few shared labelmaps as possible 1 = created labelmaps will be shared if possible without overwriting each other.&Fractional labelmap oversampling factor|1|Determines the oversampling of the reference image geometry. All segments are oversampled with the same value (value of 1 means no oversampling).&Threshold fraction|0.5|Determines the threshold that the closed surface is created at as a fractional value between 0 and 1.&Default slice thickness|0.0|Default thickness for contours if slice spacing cannot be calculated.&End capping|1|Create end cap to close surface inside contours on the top and bottom of the structure.\n0 = leave contours open on surface exterior.\n1 (default) = close surface by generating smooth end caps.\n2 = close surface by generating straight end caps.&"""
    nrrd_file_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
    nrrd_file_header['Segmentation_ReferenceImageExtentOffset'] = '0 0 0'

    nrrd.write(f'{save_path}.seg.nrrd', arr, nrrd_file_header)
    # nrrd.write(f'{save_path}.nrrd', arr, nrrd_file_header)
        

