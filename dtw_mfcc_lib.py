# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:02:25 2024

@author: Raghavan

library holding functions to calculate mfcc, dtw distances etc.
and holds all functions that will be resused across various reasearch
environments.

"""

import librosa
import numpy as np
from scipy.spatial.distance import cdist
import fnmatch
import os
import librosa.display
import glob

NUM_MFCC = 20
SAMPLE_RATE = 16000
WINDOW_SIZE = 512
HOP_SIZE = 128
NUM_QUERIES_IN_REFERENCE: int = 4
MATCH_THRESHOLD: float = 0.3  # in secs
MATCH_THERSHOLD_IN_MFCC_SETS = int((MATCH_THRESHOLD * SAMPLE_RATE) / HOP_SIZE)

def get_query_file_count():
    query_directory =  r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav"
    query_file_count = len(fnmatch.filter(os.listdir(query_directory), "*.wav"))
    num_references = int(
        (query_file_count + NUM_QUERIES_IN_REFERENCE - 1) / NUM_QUERIES_IN_REFERENCE
    )
    print("Query file Count: %d num_references %d" % (query_file_count, num_references))
    return query_file_count

"""
do_audio_align

parameters:
    1. y_src : the source audio
    2. y_dst : the destination audio
    3. mfcc_src : source mfcc set
    4. mfcc_dst : destination mfcc set

functionality:
    
    aligns the source audio with the destination audio after accounting
    for gaps and stretches. that is the audios though of similar voice
    content may have pauses and noises in between as well as the voice
    being at varying speeds. the function will remove the pauses/noises 
    and adjust the speed of the soure and destination to be the same.
algorithm:
    1. the warping path is calcuated for the global alignment 
    between source and destination.
    2. the warping path is traversed, with source on the y-axis 
    and the destination on the x-axis.
    3. if there is a horizontal on the warping math it means that
    multiple points on the x-axis (source) are mapped to a 
    single destination.
    4. that part of the source can be trimmed as it is either a 
    pause, noise or a recitation at a lower speed which has resulted
    in multiple points being mapped to a single destination point.
    5. similarly for a vertial on the warping path, the destination audio
    will be trimmed.

returns:
    the modified, aligned source and destination
    
"""

def do_audio_align(y_src, y_dst, mfcc_src, mfcc_dst):

    NUM_CONT_ALLOWED = 1
    
    last_y, last_x = -1, -1
    y_src_n = []
    y_dst_n = []
    x_st, y_st = 0,0
    x_end, y_end = -1,-1
    cont_x, cont_y = 0,0

    distances = cdist(mfcc_src.T, mfcc_dst.T)
    D, wp = librosa.sequence.dtw(C = distances)

    wp_r = wp[::-1]

    for i in range(wp_r.shape[0]):
        if (wp_r[i][0] ==  last_y):
            assert wp_r[i][1] !=  last_x
            if (x_end == -1):
                x_end = wp_r[i][1] * HOP_SIZE
            cont_y += 1
        else:
            if (cont_y >= NUM_CONT_ALLOWED):
                assert x_end != -1
                assert x_end > x_st
                y_dst_n = np.concatenate((y_dst_n, y_dst[x_st:x_end]))
                #print("Done dst conc : st %d end %d len %d next x_st %d" 
                #      %(x_st, x_end, len(y_dst_n), wp_r[i][1] * HOP_SIZE))
                x_st = wp_r[i][1] * HOP_SIZE
            cont_y = 0
            x_end = -1

        if (wp_r[i][1]==  last_x):
            assert wp_r[i][0] !=  last_y
            if (y_end == -1):
                y_end = wp_r[i][0] * HOP_SIZE
            cont_x += 1
        else:
            if (cont_x >= NUM_CONT_ALLOWED):
                assert y_end != -1
                assert y_end > y_st
                y_src_n = np.concatenate((y_src_n, y_src[y_st:y_end]))
                #print("Done src conc : st %d end %d len %d next y_st %d" 
                #      %(y_st, y_end, len(y_src_n), wp_r[i][0] * HOP_SIZE))
                y_st = wp_r[i][0] * HOP_SIZE
            cont_x = 0
            y_end = -1

        last_y = wp_r[i][0]
        last_x = wp_r[i][1]

    #print("src len %d x_st %d. dst len %d y_st %d" %(len(y_src_n), x_st, len(y_dst_n), y_st))    

    y_src_n = np.concatenate((y_src_n, y_src[y_st:len(y_src)-1]))
    y_dst_n = np.concatenate((y_dst_n, y_dst[x_st:len(y_dst)-1]))

    #print("Final src len %d dst len %d" %(len(y_src_n), len(y_dst_n)))    

    return y_src_n, y_dst_n

def load_audio(filename):
    y,sr =  librosa.load(filename, SAMPLE_RATE)
    return y

def get_mfcc(y_curr):
        return librosa.feature.mfcc(
            y=y_curr,
            sr=SAMPLE_RATE,
            n_mfcc=NUM_MFCC,
            n_fft=WINDOW_SIZE,
            hop_length=HOP_SIZE,
            center=False,
        )

def get_sample_rate():
    return SAMPLE_RATE

def get_deviation_from_slope(D, wp):
    if (D.shape[0] > D.shape[1]):
        slope = D.shape[1] / D.shape[0]
        r_idx = 0
        q_idx = 1
    else:    
        slope = D.shape[0] / D.shape[1]
        r_idx = 1
        q_idx = 0

    total_deviation = 0    
    for i in range(0, wp.shape[0]):
        total_deviation += pow(wp[i][q_idx] - wp[i][r_idx] * slope, 2)
 
    return total_deviation, total_deviation/wp.shape[0]

def calc_slope_deviation_mfcc(mfcc_src, mfcc_dst):
    distances = cdist(mfcc_src.T, mfcc_dst.T)
    D, wp = librosa.sequence.dtw(C = distances)
    return get_deviation_from_slope(D,wp)

def get_start_end_labels(filename):

    start = []
    end = []
    f_label = open(filename, "r")

    for line in f_label:
        
        words = line.split()
        s = 1
        for word in words:
            if (s == 1):
                start.append(int(float(word) * SAMPLE_RATE))
                s = 0
            else:
                end.append(int(float(word) * SAMPLE_RATE))
                break

    f_label.close()
    
    return start, end

def get_all_files_in_datatset():
    file_name = []
    query_file_count = get_query_file_count()
    directory_path = r"C:\Users\Lenovo\Desktop\dataset\ds_4min"
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    for file in txt_files:
        with open(file, 'r') as fp:
            lines = len(fp.readlines())
            assert lines == query_file_count
        base_name = os.path.basename(file)
        file_name.append(base_name[:-4])
    return file_name

def get_audio_filename(filename):
    return r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s_4.wav" % (filename)

def get_label_filename(filename):
    return r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s.txt" % (filename)

def get_sishya_app_dataset_filename():
    return "ds13"

def get_mfccs_for_audio(filename):
    
    mfcc_arr = []
    
    query_file_count = get_query_file_count()
    
    start_arr, end_arr = get_start_end_labels(get_label_filename(filename))
    assert len(start_arr) == len(end_arr) == query_file_count
    
    y_audio = load_audio(get_audio_filename(filename))
    
    for i in range(0, len(start_arr)):
        start = start_arr[i]
        end = end_arr[i]
        y_curr = y_audio[start:end]
        mfcc_arr.append(get_mfcc(y_curr))
    
    return mfcc_arr

def get_mfccs_for_app_audio():
    
    app_filename = get_sishya_app_dataset_filename()
    
    return get_mfccs_for_audio(app_filename)
