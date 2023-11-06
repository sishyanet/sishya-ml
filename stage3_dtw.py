# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 21:39:53 2023

"""

import librosa
import numpy as np
from typing import List, Tuple
import fnmatch
import os
import glob

NUM_MFCC = 20
NUM_QUERIES_IN_REFERENCE: int = 4
MATCH_THRESHOLD: float = 0.5  # in secs
query_directory: str = (
    r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav"
)

def run_iteration(file: str, l_sr: int, win_size: int, hop_size: int) -> int:

    queries_left = query_file_count
    query_file_number =  0
    mismatches = 0
    
    ref_file = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s_4.wav" % (file)
    ref_label = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\%s.txt" % (file)

    print_mismatch = 0
    
    f_label = open(ref_label, "r")
    
    y_reference, sr = librosa.load(ref_file, sr = l_sr)

    for i in range (0, num_references):

        start = []
        end = []

        c = 0
        current_number_of_queries_in_reference = min(NUM_QUERIES_IN_REFERENCE, queries_left)

        for line in f_label:

            words = line.split()
            s = 1
            for word in words:
                if (s == 1):
                    start.append((float(word)))
                    s = 0
                else:
                    end.append((float(word)))
                    break
            c += 1
            if (c == current_number_of_queries_in_reference):
                break

        start_sample = int(start[0] * l_sr)
        end_sample = int(end[current_number_of_queries_in_reference - 1] * l_sr)
        y_curr_reference = y_reference[start_sample : end_sample]
        mfcc_reference = librosa.feature.mfcc(
            y=y_curr_reference,
            sr=l_sr,
            n_mfcc=NUM_MFCC,
            n_fft=win_size,
            hop_length=hop_size,
            center=False,
        )

        for i in range (0, current_number_of_queries_in_reference):

            f_query = r"%s/%d.wav" % (query_directory, query_file_number)
            query_file_number = query_file_number + 1
            y_query, sr = librosa.load(f_query, sr = l_sr)
            mfcc_query = librosa.feature.mfcc(
                y=y_query,
                sr=l_sr,
                n_mfcc=NUM_MFCC,
                n_fft=win_size,
                hop_length=hop_size,
                center=False,
            )

            D1, wp1 = librosa.sequence.dtw(mfcc_query, mfcc_reference, subseq=True,\
                                         step_sizes_sigma=np.array([[1,1], [1,0], [1,1]]))
            start_index1 = wp1[-1][1]
            end_index1 = wp1[0][1]

            D2, wp2 = librosa.sequence.dtw(mfcc_query, mfcc_reference, subseq=True,\
                                         step_sizes_sigma=np.array([[1,1], [0,1], [1,1]]))
            start_index2 = wp2[-1][1]
            end_index2 = wp2[0][1]
            
            expected_start = start[i] - start[0]
            expected_end = end[i] - start[0]

            if (
                abs(expected_start - ((start_index1 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs(expected_end - ((end_index1 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs(expected_start - ((start_index2 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs(expected_end - ((end_index2 * hop_size) / l_sr))
                > MATCH_THRESHOLD
            ):
                if (print_mismatch):
                    print("*** MISMATCH ***")
                    print("Query file %d Ref start %d i %d.\
                        Expected start %d end %d. Returned start %d, %d end %d, %d"\
                        % (query_file_number, start_sample, i, expected_start * l_sr, \
                            expected_end * l_sr, (start_index1 * hop_size) , \
                            (start_index2 * hop_size) , (end_index1 * hop_size), (end_index2 * hop_size)))
                     
                mismatches += 1

        queries_left = queries_left - current_number_of_queries_in_reference

    f_label.close()

    assert queries_left == 0
    assert query_file_number == query_file_count
    
    return mismatches    

def main():

    global query_file_count
    query_file_count = len(fnmatch.filter(os.listdir(query_directory), "*.wav"))
    global num_references
    num_references = int(
        (query_file_count + NUM_QUERIES_IN_REFERENCE - 1) / NUM_QUERIES_IN_REFERENCE
    )
    print("Query file Count: %d num_references %d" % (query_file_count, num_references))

    file_name = []
    directory_path = r"C:\Users\Lenovo\Desktop\dataset\ds_4min"
    txt_files = glob.glob(os.path.join(directory_path, "*.txt"))
    for file in txt_files:
        base_name = os.path.basename(file)
        file_name.append(base_name[:-4])
    
    sr, win, hop = 16000, 512, 128
    print("SR %d window size %d hop size %d" %(sr, win, hop))
    print("File\tMismatches")
    for i in range(0, len(file_name)):
        mismatches = run_iteration(file_name[i], sr, win, hop)
        print("%s\t%d" %(file_name[i], mismatches))

if __name__ == "__main__":
    main()
    