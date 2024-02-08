# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:08:00 2024

@author: Raghavan
"""

import dtw_mfcc_lib as dml

def calc_slope_deviation(mfcc_app_arr, mfcc_audio_arr):
    
    assert len(mfcc_app_arr) == len(mfcc_audio_arr)
    
    for i in range(0, len(mfcc_app_arr)):
        mfcc_app = mfcc_app_arr[i]
        mfcc_audio = mfcc_audio_arr[i]
        deviation, nrl_deviation = dml.calc_slope_deviation_mfcc(mfcc_app, mfcc_audio)
        print("[%d] deviation %.2f nrl %.2f" %(i, deviation, nrl_deviation))

def main():
    mfcc_app = dml.get_mfccs_for_app_audio()
    file_name = dml.get_all_files_in_datatset()
    for i in range (0, len(file_name)):
        if (file_name[i] == dml.get_sishya_app_dataset_filename()):
            continue
        mfcc_audio = dml.get_mfccs_for_audio(file_name[i])
        print("slope deviation for %s" %(file_name[i]))
        calc_slope_deviation(mfcc_app, mfcc_audio)

if __name__ == "__main__":
    main()
    