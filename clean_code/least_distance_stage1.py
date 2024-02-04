# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 22:38:53 2023

@author: Raghavan

This code sees if the mfcc's along with least distance algorithm 
can be used to match a small part of an audio within a large part.

This is stage1 where the comparision is done in the time domain.
That is a small duration of the audio is used as query and the larger
duration is used as reference.

In stage0 the comparision was done at the mfcc domain, meaning a small
subset of mfcc's was queried against a superset of mfcc's

"""

import librosa
import least_distance as ld

N_MFCC: int = 13
REFERENCE_DURATION: int = 5  # in secs
QUERY_DURATION: int = 2  # in secs
QUERY_OFFSET_DURATION: int = 1  # in secs
hop_samples: int = 512
window_samples: int = 2048
NUM_WINDOWS_RANGE = 2

check_with_dtw = 0

"""
The audio is split into references,of duration REFERENCE_DURATION secs.
In each reference (REFERENCE_DURATION - QUERY_DURATION)
/ QUERY_OFFSET_DURATION queries are formed. 

If the REFERENCE_DURATION is 5 secs and QUERY_DURATION is 2 sec and 
QUERY_OFFSET_DURATION is 1 sec, then queries from 0-2sec, 1-3, 2-4, 3-5
are formed.
"""

def main():
    
    file: str = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\ds1_4.wav"
    y, sr = librosa.load(file)
        
    reference_samples: int = int(sr * REFERENCE_DURATION)
    query_samples: int = int(sr * QUERY_DURATION)
    query_offset_samples: int = int(sr * QUERY_OFFSET_DURATION)
    
    print(
        "Ref samples %d q %d q off %d hop %d win %d"
        % (
            reference_samples,
            query_samples,
            query_offset_samples,
            hop_samples,
            window_samples,
        )
    )
    
    # doing len(y) - reference_samples to leave out the last part, which maybe small enough to be lesser than n_fft
    for i in range(0, len(y) - reference_samples, reference_samples):
        y_reference = y[i : i + reference_samples]
    
        mfcc_reference = librosa.feature.mfcc(
            y=y_reference,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=window_samples,
            hop_length=hop_samples,
            center=False,
        )
    
        for j in range(i, i + reference_samples - query_samples, query_offset_samples):
            y_query = y[j : j + query_samples]
    
            mfcc_query = librosa.feature.mfcc(
                y=y_query,
                sr=sr,
                n_mfcc=N_MFCC,
                n_fft=window_samples,
                hop_length=hop_samples,
                center=False,
            )
    
            if (check_with_dtw == 1):
                start_index, end_index, min_dist = ld.least_distance_mfcc_dtw(mfcc_reference.T, mfcc_query.T)
            else:
                start_index, end_index, min_dist = ld.least_distance_mfcc(mfcc_reference.T, mfcc_query.T)
    
            expected_start, expected_end = (j - i), (j - i) + query_samples
    
            """
            there will not be an exact match since the comparison is in the
            time domain. For eg: if the query is from 2-4 secs, then the
            start and the end of the query will not be aligned with the 
            hops within the reference. Hence seeing if the matches are
            within a range of NUM_WINDOWS_RANGE number of windows.
            """
    
            start_range_begin = expected_start - (window_samples * NUM_WINDOWS_RANGE)
            start_range_end = expected_start + (window_samples * NUM_WINDOWS_RANGE)
    
            end_range_begin = expected_end - (window_samples * NUM_WINDOWS_RANGE)
            end_range_end = expected_end + (window_samples * NUM_WINDOWS_RANGE)
    
            if (start_index * hop_samples) not in range(
                start_range_begin, start_range_end
            ) or (end_index * hop_samples) not in range(end_range_begin, end_range_end):
                print(
                    "*** i %d j %d exp st %d [diff %d] end %d [diff %d]"
                    % (
                        i,
                        j,
                        expected_start,
                        abs(start_index * hop_samples - expected_start),
                        expected_end,
                        abs(end_index * hop_samples - expected_end),
                    )
                )
    
                print(
                    "start = %d(sec %f sample %d) end = %d (sec %f sample %d)"
                    % (
                        start_index,
                        (start_index * hop_samples) / sr,
                        start_index * hop_samples,
                        end_index,
                        (end_index * hop_samples) / sr,
                        end_index * hop_samples,
                    )
                )
                print(
                    "Look for %d to %d against %d to %d"
                    % (
                        j,
                        j + query_samples,
                        i + (start_index * hop_samples),
                        i + (end_index * hop_samples),
                    )
                )
    
    print("DONE :")

if __name__ == '__main__':
    main()
