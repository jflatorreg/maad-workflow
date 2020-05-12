#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Soundclim Utilities
Module with functions useful to compile datasets for SoundClim

VERSION 0.2
Created on Feb 2019 (version 0.1)
Updated on May 2019 (version 0.2)
Updated on July 2019 (version 0.3): 
    bug on batch_feature_rois solved by removing db_range parameter
@author: jseb.ulloa@gmail.com
"""

import pandas as pd
import numpy as np
from os import listdir
import time
import joblib
import matplotlib.pyplot as plt
from sklearn import manifold, preprocessing
from librosa.core import resample
from os import listdir
from maad.rois import find_rois_cwt
import time
from maad import sound
from maad.features import shape_features, centroid, opt_shape_presets, compute_rois_features
from maad.util import format_rois, rois_to_imblobs, normalize_2d

def visual_rois(xdata, idx_highlight=-1, perplexity=50, plot=True):
    """
    Computes a 2D visualization of the rois shape features space with t-SNE

    Note: the function only selects shape features since they are homogeneous
    
    Parameters:
    ----------
        xdata: pandas dataframe
            A data frame with shape ('shp') columns
        idx_highlight: ndarray
            Boolean array with indices to be highlighted on the image
    Returns:
    -------
        tsne: float
            Transformed data by tsne algorithm in a 2D space
            
    """    

    # select features
    X = xdata
    # compute tsne
    time_start = time.clock()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, 
                     verbose=1, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    time_elapsed = (time.clock() - time_start)
    print('\nShape features transformed with t-SNE in', np.round(time_elapsed,3), 's')
    if plot is True:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=[8,8])
        ax.scatter(Y[:, 0], Y[:, 1], s = 10, alpha=0.5, c='#1f77b4')
        ax.set_xlabel('x-tsne')
        ax.set_ylabel('y-tsne')
        if idx_highlight is not -1:
            ax.scatter(Y[idx_highlight, 0], Y[idx_highlight, 1], s = 20, 
                        alpha=0.5, color='darkorange')
        else: 
            pass
    else:
        pass
    
    return Y
    
def batch_find_rois(flist, params_detections, path_audio):
    """
    Exports features saved as joblib into a csv file readable by R and other 
    programs. The joblib file should be computed using the 
    
    Parameters:
    ----------
        params_detection: dict
            Dictionary with the basic parameters to feed find_rois: 
            'flims', 'tlen', and 'th'.
        path_flist : str
            Path to a *.txt file with the list of audio filenames to process
        path_audio : str
            Path to the place were the dataset of audio files are stored

    Returns:
    -------
        Saves a joblib file to disk. Does not return any variable
            
    """    
    # load parameters
    flims = params_detections['flims']
    tlen = params_detections['tlen']
    th = params_detections['th']
    
    detections = list()
    for idx, fname in enumerate(flist['fname']):
        print(idx+1, '/', len(flist), fname)
        s, fs = sound.load(path_audio+fname)
        rois = find_rois_cwt(s, fs, flims, tlen, th)
        if not rois.empty:    
            # filter rois shorter than 25% of tlen
            idx_rm = (rois.max_t - rois.min_t) < tlen*0.25
            rois.drop(index=np.where(idx_rm)[0], inplace=True)
            rois.reset_index(inplace=True, drop=True)
        else:
            pass
        # save to list
        detections.append({'fname':fname, 'rois':rois})
    
    info_detections = {'detections': detections, 'parameters': params_detections}
    return info_detections


def joblib_features_to_csv(path_data, path_save):
    """
    Exports features saved as joblib into a csv file readable by R and other 
    programs. The joblib file should be computed using the 
    
    Parameters:
    ----------
        path_data : str
            string indicating the path to the joblib file
        path_save : str
            string with the file name to save the csv

    Returns:
    -------
        Saves a file to disk. Does not return any variable
            
    """
    
    info_features = joblib.load(path_data)
    features = info_features['features']
    
    # get xdata from object features
    for idx, file in enumerate(features):
        fname = file['fname']
        aux_df = file['features']
        aux_df['fname'] = fname
        if idx is 0:
            xdata = aux_df
        else:
            xdata = pd.concat([xdata, aux_df], axis=0, sort=False)
    
    xdata.reset_index(drop=True, inplace=True)
    xdata.to_csv(path_save, index=False, sep=',')
    

def features_to_csv(features_data):
    """
    Exports features object into a csv file
    
    Parameters:
    ----------
        features_data : str
            features object computed with batch_feature_rois
        path_save : str
            string with the file name to save the csv

    Returns:
    -------
        Saves a file to disk. Does not return any variable
            
    """
    
    features = features_data['features']
    
    # get xdata from object features
    for idx, file in enumerate(features):
        fname = file['fname']
        aux_df = file['features']
        aux_df['fname'] = fname
        if idx == 0:
            xdata = aux_df
        else:
            xdata = pd.concat([xdata, aux_df], axis=0, sort=False)
    
    xdata.reset_index(drop=True, inplace=True)
    return xdata


def batch_feature_rois(rois_list, params_features, path_audio):
    """
    Computes features for a list of files
    
    Parameters:
    ----------
        params_features: dict
            Dictionary with the basic parameters to feed find_rois: 
            'flims', 'tlen', and 'th'.
        path_flist : str
            Path to a *.txt file with the list of audio filenames to process
        path_audio : str
            Path to the place were the dataset of audio files are stored
        path_save : str
            Path with the file name to save the csv

    Returns:
    -------
        info_features: dic
            Dictionary with features and all the parameters used to compute the features.
            Included keys: features, parameters_df, opt_shape, opt_spectro
            
    """    
    ## TODO: when the time limits are too short, the function has problems
    # load parameters
    flims = params_features['flims']
    opt_spec = params_features['opt_spec']
    opt_shape = opt_shape_presets(params_features['opt_shape_str'])

    # load detection data
    
    features = []
    for idx, file in enumerate(rois_list):   
        # unpack file values
        fname = file['fname']
        rois_tf = file['rois']
        print(idx+1, '/', len(rois_list), fname)    
        
        if rois_tf.empty:
            print('< No detection on file >')
            features.append({'fname':fname, 'features': pd.DataFrame()})
        else:
            # load materials: sound, spectrogram
            s, fs = sound.load(path_audio+fname)
            im, dt, df, ext = sound.spectrogram(s, fs, nperseg=opt_spec['nperseg'], 
                                                overlap=opt_spec['overlap'], fcrop=flims, 
                                                rescale=False, db_range=opt_spec['db_range'])
            
            # format rois to bbox
            ts = np.arange(ext[0], ext[1], dt)
            f = np.arange(ext[2],ext[3]+df,df)
            rois_bbox = format_rois(rois_tf, ts, f, fmt='bbox')
                
            # roi to image blob
            im_blobs = rois_to_imblobs(np.zeros(im.shape), rois_bbox)
            
            # get features: shape, center frequency
            im = normalize_2d(im, 0, 1)
            bbox, params, shape = shape_features(im, im_blobs, resolution='custom', 
                                                 opt_shape=opt_shape)
            _, cent = centroid(im, im_blobs)
            cent['frequency']= f[round(cent.y).astype(int)]  # y values to frequency
            
            # format rois to time-frequency
            rois_out = format_rois(bbox, ts, f, fmt='tf')
            
            # combine into a single df
            aux_df = pd.concat([rois_out, shape, cent.frequency], axis=1)
            #        aux_df['fname'] = fname
            features.append({'fname':fname, 'features': aux_df})
    
    
    # Arranges the data into a dictionary
    info_features = {'features': features,
                     'parameters_df': params,
                     'opt_shape': opt_shape,
                     'opt_spectro': opt_spec}
    return info_features


def batch_predict_rois(flist, tuned_clfs, params, path_audio_db='./'):
    """
    Predict the labels of rois in a list of audio files. 
    
    Parameters
    ----------
    flist: pandas DataFrame
        list of audio filenames to be analysed. Column name must be 'fname'
    tuned_clfs: dict
        data structure with tuned classifiers by grid search or random search
    params: dict
        data structure with the same parameters used to train the classifiers.
        Keys to be included: 'sample_rate_wav', 'flims', 'tlen', 'th', 
        'opt_spec', 'opt_shape_str'
    path_audio_db: str, default current directory
        path pointing to the directory where the audio files are located. 
        Note that all files in flist must be in the same directory
    
    Returns
    -------
    predictions: dict
        data structure with name of audio files as keys. Each element in the
        dictionary has a DataFrame with predictions for every region interest
        found. Predictions are given as probabilities for three different 
        classifiers, namely Random Forest ('rf'), Adaboost ('adb') and Support
        Vector Machines ('svm').
    
    """
    t_start = time.time() # compute processing time
    # Load params and variables
    clf_svm = tuned_clfs['svm'].best_estimator_
    clf_rf = tuned_clfs['rf'].best_estimator_
    clf_adb = tuned_clfs['adb'].best_estimator_
    flims = params['flims']
    tlen = params['tlen']
    th = params['th']
    opt_spec = params['opt_spec']
    opt_shape = opt_shape_presets(params['opt_shape_str'])
    sample_rate_std = params['sample_rate_wav']
    
    # Batch: compute rois, features and predict through files
    predictions = dict()
    for idx, fname in enumerate(flist['fname']):
        print(idx+1, '/', len(flist), fname)
        # fname = flist['fname'][0]
        s, fs = sound.load(path_audio_db+fname)
        # Check sampling frequency on file
        if fs==sample_rate_std:
            pass
        else:
            print('Warning: sample rate mismatch, resampling audio file to standard', 
                  sample_rate_std, 'Hz')
            s = resample(s, fs, sample_rate_std, res_type='kaiser_fast')
            fs = sample_rate_std
            
        rois = find_rois_cwt(s, fs, flims, tlen, th)    
        if rois.empty:
            #print('< No detection on file >')
            predictions[fname] = -1
        else:
            # filter rois shorter than 25% of tlen
            idx_rm = (rois.max_t - rois.min_t) < tlen*0.25
            rois.drop(index=np.where(idx_rm)[0], inplace=True)
            rois.reset_index(inplace=True, drop=True)
            if rois.empty:
                print('< No detection on file >')
                predictions[fname] = -1
            else:        
                # compute features
                rois_features = compute_rois_features(s, fs, rois, opt_spec, opt_shape, flims)
                # predict
                X = rois_features.loc[:,rois_features.columns.str.startswith('shp')]
                #X['frequency'] = preprocessing.scale(X['frequency'])  # new! scale frequency
                pred_rf = pd.DataFrame(data=clf_rf.predict_proba(X), 
                                       columns=[s + '_rf' for s in clf_rf.classes_.astype('str')])
                pred_adb = pd.DataFrame(data=clf_adb.predict_proba(X), 
                                        columns=[s + '_adb' for s in clf_adb.classes_.astype('str')])
                pred_svm = pd.DataFrame(data=clf_svm.predict_proba(X), 
                                        columns=[s + '_svm' for s in clf_svm.classes_.astype('str')])
                # save to variable
                pred_proba_file = pd.concat([rois, pred_rf, pred_adb, pred_svm], axis=1)
                predictions[fname] = pred_proba_file
    
    t_stop = time.time() # compute processing time
    print('Batch process completed. Processing time: ', np.round(t_stop - t_start,2),'s')
    return predictions

def listdir_pattern(path_dir, ends_with=None):
    """
    Wraper function from os.listdir to include a filter to search for patterns
    
    Parameters
    ----------
        path_dir: str
            path to directory
        ends_with: str
            pattern to search for at the end of the filename
    Returns
    -------
    """
    flist = listdir(path_dir)
    
    new_list = []
    for names in flist:
         if names.endswith(ends_with):
            new_list.append(names)
    return new_list

def read_audacity_annot (audacity_filename):
    """
    Read audacity annotations file (or labeling file) and return a Pandas Dataframe
    with the bounding box and the label of each region of interest (ROI)
    
    Parameters
    ----------
    audacity_filename : str
        Path to the audacity file

    Returns
    -------
    tab_out : pandas DataFrame 
        Colormap type used by matplotlib
    
    References
    ----------
    https://manual.audacityteam.org/man/label_tracks.html   
    """
    # read file with tab delimiter
    tab_in = pd.read_csv(audacity_filename, delimiter='\t', header=None)
    
    # arrange data
    t_info = tab_in.loc[np.arange(0,len(tab_in),2),:]
    t_info = t_info.rename(index=str, columns={0: 'min_t', 1: 'max_t', 2:'label'})
    t_info = t_info.reset_index(drop=True)
    
    f_info = tab_in.loc[np.arange(1,len(tab_in)+1,2),:]
    f_info = f_info.rename(index=str, columns={0: 'slash', 1: 'min_f', 2:'max_f'})
    f_info = f_info.reset_index(drop=True)
    
    # return dataframe
    tab_out = pd.concat([t_info['label'].astype('str'), 
                         t_info['min_t'].astype('float32'), 
                         f_info['min_f'].astype('float32'), 
                         t_info['max_t'].astype('float32'), 
                         f_info['max_f'].astype('float32')],  axis=1)

    return tab_out

def predictions_to_df(predictions, clf_lab_list):
    """
    Parameters
    ----------
    predictions : dict
        Prediction element from the function batch_predict_rois
    clf_lab_list : list
        Names of classifiers to get 

    Returns
    -------
    res : pandas DataFrame
        Highest score for each classifier

    """
    res = pd.DataFrame()
    for clf_lab in clf_lab_list:
        pred_file = dict()
        for fname, pred in predictions.items():
            if type(pred) is int:  # case where no ROIs were found
                pred_file[fname] = 0
            else:
                # compute argmax on labels
                positive_max = np.amax(pred[clf_lab].max())
                n_high_prob = (pred.loc[:,clf_lab] > 0.5).sum()
                pred_file[fname] = [positive_max, n_high_prob]
            
        pred_file = pd.DataFrame(data=pred_file).transpose()
        pred_file = pred_file.reset_index()
        pred_file.columns = ['fname',clf_lab,'n_high_proba']
        res = pd.concat([res, pred_file[clf_lab]], axis=1)
    
    res['fname'] = pred_file['fname']
    return res

def format_trainds(df, flims, wl, path_audio):
    """
    Arranges all the training data into a dictionary for easy and compact access.
    
    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with information on the regions of interest to be arranged.
        The DataFrame must have the columns: fname, min_t, max_t.
    flims : tuple or list
        Minimum and maximum frequency limits of the band pass filter. This
        is used to filter unwanted sounds and improve the manual analysis.
    wl : int or float
        Window length (in seconds) of each region of intrest. While the regions 
        have a specified duration, with this argument it is possible to increase
        the window of observation, allowing to have a wider context to analyse 
        the audio. Recomended minimum 2 seconds.
    path_audio : str
        Path to the directory where all the raw audio files are stored

    Returns
    -------
    train_data : dict
        A dictionary with the keys: roi_info, shape_features, label, audio, segments and maad_label

    """
    print('Aligning ROIs, number of observations:', len(df))
    df['tlen'] = df.max_t - df.min_t
    audiolist = list()
    for idx, roi in df.iterrows():
        fname_wav = path_audio + roi.fname
        # define tlimits with window length
        length = roi.max_t - roi.min_t
        tlims = ((roi.min_t + length/2) - wl/2, (roi.min_t + length/2) + wl/2)
        s, fs = sound.load(fname_wav)
        s = sound.select_bandwidth(s, fs, lfc=flims[0], hfc=flims[1])
        # #normalize?
        rec_length = len(s)/fs
        
        # if time limits are outside the recording, add silence
        if tlims[1] > rec_length:
            # add silence at end
            sil_len = tlims[1] - rec_length
            silence = np.zeros(int(sil_len*fs))
            s_roi = np.concatenate([s[int(tlims[0]*fs):], silence])
        
        elif tlims[0] < 0:
            # add silence at begin
            sil_len = abs(tlims[0])
            silence = np.zeros(int(sil_len*fs))
            s_roi = np.concatenate([silence, s[0:int(tlims[1]*fs)]])
            
        else:
            s_roi = s[int(tlims[0]*fs):int(tlims[1]*fs)]

        audiolist.append(s_roi.copy())
        
    ## write segments for manual annotations
    onset = (wl/2) - (df.tlen/2)
    offset = (wl/2) + (df.tlen/2)
    seg = pd.DataFrame({'onset': onset, 'offset': offset})
    seg['label'] = 'NA'
    
    ## assign to object and save
    train_data = dict()
    idx_features = df.columns.str.startswith('shp') | (df.columns=='frequency')
    train_data['roi_info'] = df[['fname','min_t','max_t','min_f','max_f']]
    train_data['shape_features'] = df.loc[:,idx_features] 
    train_data['label'] = seg.label
    train_data['audio'] = audiolist
    train_data['segments'] = seg[['onset','offset']]
    train_data['maad_label'] = df.cluster
    return train_data