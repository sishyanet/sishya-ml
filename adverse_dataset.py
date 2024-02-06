# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 13:56:33 2024

@author: Raghavan
"""

import dtw_mfcc_lib as dml
import IPython
from IPython.display import Audio

def collect_adverse_dataset(filename):
    
    sr = dml.get_sample_rate()
    
    query_file_count = dml.get_query_file_count()
        
    dst_file = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s_4.wav" % (filename)
    dst_label = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s.txt" % (filename)
        
    f_label = open(dst_label, "r")
    
    y_dst = dml.load_audio(dst_file)
    
    print("Collecting adverse dataset for %s" %(filename))
    start_arr , end_arr = dml.get_start_end_labels(dst_label)
    
    assert len(start_arr) == len(end_arr) == query_file_count

    for dataset_idx in range(0, len(start_arr)):

        src_file = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav\%d.wav" %(dataset_idx)
        y_src = dml.load_audio(src_file)
        mfcc_src = dml.get_mfcc(y_src)
        
        start = start_arr[dataset_idx]
        end = end_arr[dataset_idx]
        
        y_dst_curr = y_dst[start:end]
        mfcc_dst = dml.get_mfcc(y_dst_curr)
        
        IPython.display.display(Audio(data=y_src, rate=sr))
        IPython.display.display(Audio(data=y_dst_curr, rate=sr))
        print("Original src and dst audio %d" %(dataset_idx))

        y_src_n, y_dst_n = dml.do_audio_align(y_src, y_dst_curr, mfcc_src, mfcc_dst)
        
        min_len = min(len(y_src_n), len(y_dst_n))
        y_comb_n = y_src_n[0:min_len] + y_dst_n[0:min_len]
        IPython.display.display(Audio(data=y_comb_n, rate=sr))   
        print("Combined modified audio %d" %(dataset_idx))
    
    f_label.close()

def main():    
    file_name = dml.get_all_files_in_datatset()
    for i in range(0, len(file_name)):
        if (file_name[i] != dml.get_sishya_app_dataset_filename()):
            collect_adverse_dataset(file_name[i])

if __name__ == "__main__":
    main()

