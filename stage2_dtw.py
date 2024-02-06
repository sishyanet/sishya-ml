# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:47:20 2023

"""

import librosa
import numpy as np
from scipy.spatial.distance import cdist
from math import inf
from typing import List, Tuple
import fnmatch
import os

'''
Takes a collection of queries to form a reference.
Modifies the query by pitch and time.
Finds the query within the reference.

'''

NUM_QUERIES_IN_REFERENCE: int = 10
MATCH_THRESHOLD: float = 0.5  # in secs
query_directory: str = (
    r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav"
)

def run_iteration(l_sr: int, l_n_mfcc: int, win_size: int, hop_size: int, p_shift: float, t_shift: float) -> int:

    y_reference_all = []
    mfcc_reference_all = []
    y_query_array_all = []

    queries_left: int = query_file_count
    query_file_number: int = 0
    
    # the queries (76 currently) are coalesced to form a reference.
    # If 10 are coalesced to one reference, then there would be 8 referenes
    
    for i in range(0, num_references):

        current_number_of_queries_in_reference = min(NUM_QUERIES_IN_REFERENCE, queries_left)
        
        y_reference = []
        y_query_array = []

        for j in range(0, current_number_of_queries_in_reference):

            f = r"%s/%d.wav" % (query_directory, query_file_number)
            query_file_number = query_file_number + 1

            y_query, sr = librosa.load(f, sr = l_sr)
            y_reference = np.concatenate((y_reference, y_query))
            y_query_array.append(y_query)

        assert len(y_reference) == sum(len(x) for x in y_query_array)
        
        y_reference_all.append(y_reference)
        y_query_array_all.append(y_query_array)

        mfcc_reference = librosa.feature.mfcc(
            y=y_reference,
            sr=l_sr,
            n_mfcc=l_n_mfcc,
            n_fft=win_size,
            hop_length=hop_size,
            center=False,
        )
        mfcc_reference_all.append(mfcc_reference)        

        queries_left = queries_left - current_number_of_queries_in_reference

    queries_left: int = query_file_count
    
    mismatches: int = 0

    early_exit = False
    print_mismatch = False

    for i in range(0, num_references):

        current_number_of_queries_in_reference = min(NUM_QUERIES_IN_REFERENCE, queries_left)

        y_reference = y_reference_all[i]
        y_query_array = y_query_array_all[i]

        mfcc_reference = mfcc_reference_all[i]

        current_query_start, current_query_end = 0, 0

        for j in range(0, current_number_of_queries_in_reference):

            if j != 0:
                current_query_start = current_query_end + 1

            current_query_end = current_query_start + len(y_query_array[j]) - 1

            modified_query = librosa.effects.pitch_shift(
                y=y_query_array[j], sr=l_sr, n_steps=p_shift)

            modified_query = librosa.effects.time_stretch(y=modified_query, rate=t_shift)

            mfcc_query = librosa.feature.mfcc(
                y=modified_query,
                sr=l_sr,
                n_mfcc=l_n_mfcc,
                n_fft=win_size,
                hop_length=hop_size,
                center=False,
            )

            '''
            Doing two alignments.
            1. Only diagonal and horizontal.
            2. Only diagonal and vertical.
            The idea is that when doing a local alignment DTW, for
            our usecase, in some cases, the query gets aligned to a
            small part of a reference, not from the expected region.
            
            The x-axis represent the reference, y-axis the query. If a small
            part of the reference is similar to the query, then the alignment
            would be mostly vertical, with lot of query elements being
            matched to small number of reference elements. And such
            a match will not be in the expected region, where an almost
            diagonal movement is expected.
            To prevent such a mismatch, the alignment movements are restricted
            to only diagonal and horizontal.
            But with such a restriction on vertical movement, when the query
            is a time stretched version, there will be case where multiple
            query elements are needed to be mapped to fewer reference elements
            and this will need vertical movements.
            Hence an alignment is done with just diagonal and vertical
            movements to satisfy this case.
            '''
            D1, wp1 = librosa.sequence.dtw(mfcc_query, mfcc_reference, subseq=True,\
                                         step_sizes_sigma=np.array([[1,1], [1,0], [1,1]]))
            start_index1 = wp1[-1][1]
            end_index1 = wp1[0][1]

            D2, wp2 = librosa.sequence.dtw(mfcc_query, mfcc_reference, subseq=True,\
                                         step_sizes_sigma=np.array([[1,1], [0,1], [1,1]]))
            start_index2 = wp2[-1][1]
            end_index2 = wp2[0][1]

            if (
                abs((current_query_start / l_sr) - ((start_index1 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs((current_query_end / l_sr) - ((end_index1 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs((current_query_start / l_sr) - ((start_index2 * hop_size) / l_sr))
                > MATCH_THRESHOLD
                and abs((current_query_end / l_sr) - ((end_index2 * hop_size) / l_sr))
                > MATCH_THRESHOLD
            ):
                if (print_mismatch):
                    print("*** MISMATCH ***")
                    print(
                        "i %d j %d. Expected start %d end %d. Returned start %d, %d end %d, %d"
                        % (
                            i,
                            j,
                            current_query_start ,
                            current_query_end ,
                            (start_index1 * hop_size) ,
                            (start_index2 * hop_size) ,
                            (end_index1 * hop_size),
                            (end_index2 * hop_size),
                        )
                    )
                mismatches += 1
                if (early_exit):
                    break

        assert current_query_end == len(y_reference) - 1

        queries_left = queries_left - current_number_of_queries_in_reference

    assert queries_left == 0

    return mismatches

def run_diff_parameters(s_rate: int, num_mfcc: int, win_samples: int, hops: int):
    print("SR %d NUM_FEATURES %d window %d hop %d" %(s_rate, num_mfcc, win_samples, hops))
    p_s = -10
    for i in range(0, 100):
        print("Pitch Shift %.1f" %(p_s))
        t_s = 0.8
        print("Time stretch\tMismatches")
        for j in range(0,5):
            mismatch_count = run_iteration(s_rate, num_mfcc, win_samples, hops, p_s, t_s)
            print("%.1f\t%d" %(t_s, mismatch_count))
            t_s = t_s + 0.1
        p_s = p_s + 0.2

def main():
    #The directory holds segmented sishya app files. The files will be
    #concatenated to form the reference.
    global query_file_count
    query_file_count = len(fnmatch.filter(os.listdir(query_directory), "*.wav"))
    global num_references
    num_references = int(
        (query_file_count + NUM_QUERIES_IN_REFERENCE - 1) / NUM_QUERIES_IN_REFERENCE
    )
    print("Query file Count: %d num_references %d" % (query_file_count, num_references))

    run_diff_parameters(8000, 20, 2048, 512)
    
if __name__ == "__main__":
    main()
    