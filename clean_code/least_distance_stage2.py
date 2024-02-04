# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 22:38:53 2023

@author: Raghavan

This code sees if the mfcc's along with least distance algorithm 
can be used to match a small part of an audio within a large part,
when the small part is modified synthetically (using inbuilt librosa
functions that do time stretch and pitch shift).

This is stage2. Stage1 was where the comparision was done without the
synthetic mofidifcations.
That is a small duration of the audio is used as query and the larger
duration is used as reference.

In stage0 the comparision was done at the mfcc domain, meaning a small
subset of mfcc's was queried against a superset of mfcc's

If PITCH_SHIFT is on, the pitch will be shifted by a factor of
LIBROSA_SEMITONE_STEP. A negative LIBROSA_SEMITONE_STEP will mean that the
pitch has been lowered, a positive value will mean that the pitch has been
made higher. If TIME_STRETCH is on, for a positive LIBROSA_TIME_STRETCH
value, the audio will be made faster and correspondingly altered for a 
negative LIBROSA_TIME_STRETCH value.

The reference will be formed by combining NUM_QUERIES_IN_REFERENCE chunks.
The query_directory is the folder that holds the sishya app audio clips 
(quaters of a sentence). There are 76 clips in the current directory.
If NUM_QUERIES_IN_REFERENCE is 10, then every 10 of the 76 would be combined
to form a reference. The last 6 would form the final reference.
Then every chunk within the reference is taken for modification.
The modified audio is fed as the query to the algorithm. The algorithm is 
expected to match the query within the reference at its actual location.
A match is when the algorithm returned start and end lie within 
MATCH_THRESHOLD.

"""

import librosa
import numpy as np
import least_distance as ld
import fnmatch
import os

NUM_FEATURES: int = 36
hop_samples: int = 2048
window_samples: int = 4096
NUM_QUERIES_IN_REFERENCE: int = 10
LIBROSA_SEMITONE_STEP: float = 0.3
LIBROSA_TIME_STRETCH: float = 1.1
MATCH_THRESHOLD: float = 0.5  # in secs
PITCH_SHIFT: int = 1
TIME_STRETCH: int = 1
DO_MFCC_FEATURE: int = 1
query_directory: str = (
    r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav"
)

"""

modify_query_librosa:

parameters:
1. y_query : the query audio to be modified
2. sr : the sample rate

functionality:
If PITCH_SHIFT, then pitch shift is done. Then a time stretch is done 
on the pitch shifted audio.

returns:
y_query: modified query

"""

def modify_query_librosa(y_query: np.ndarray, sr: int) -> np.ndarray:
    if PITCH_SHIFT:
        y_query = librosa.effects.pitch_shift(
            y=y_query, sr=sr, n_steps=LIBROSA_SEMITONE_STEP
        )
    if TIME_STRETCH:
        y_query = librosa.effects.time_stretch(y=y_query, rate=LIBROSA_TIME_STRETCH)
    return y_query

"""

do_feature_extraction:

parameters:
1. y_data : the audio for which the features need to be extracted
2. sr : the sample rate

returns:
The MFCC or chroma feature vector.

"""

def do_feature_extraction(y_data: np.ndarray, sr: int) -> np.ndarray:

    if DO_MFCC_FEATURE:
        return librosa.feature.mfcc(
            y=y_data,
            sr=sr,
            n_mfcc=NUM_FEATURES,
            n_fft=window_samples,
            hop_length=hop_samples,
            center=False,
        )

    return librosa.feature.chroma_stft(
        y=y_data,
        sr=sr,
        n_chroma=NUM_FEATURES,
        n_fft=window_samples,
        hop_length=hop_samples,
        center=False,
    )

def main():
    #The directory holds segmented sishya app files. The files will be
    #concatenated to form the reference.
    query_file_count: int = len(fnmatch.filter(os.listdir(query_directory), "*.wav"))
    num_references: int = int(
        (query_file_count + NUM_QUERIES_IN_REFERENCE - 1) / NUM_QUERIES_IN_REFERENCE
    )
    print("Query file Count: %d num_references %d" % (query_file_count, num_references))
    
    queries_left: int = query_file_count
    query_file_number: int = 0
    mismatches: int = 0
    
    for i in range(0, num_references):
    
        current_number_of_queries_in_reference = min(NUM_QUERIES_IN_REFERENCE, queries_left)
    
        y_reference = []
        y_query_array = []
    
        for j in range(0, current_number_of_queries_in_reference):
    
            f = r"%s/%d.wav" % (query_directory, query_file_number)
            query_file_number = query_file_number + 1
    
            y_query, sr = librosa.load(f)
            y_reference = np.concatenate((y_reference, y_query))
            y_query_array.append(y_query)
    
        assert len(y_reference) == sum(len(x) for x in y_query_array)
    
        mfcc_reference = do_feature_extraction(y_reference, sr)
    
        current_query_start, current_query_end = 0, 0
    
        for j in range(0, current_number_of_queries_in_reference):
    
            if j != 0:
                current_query_start = current_query_end + 1
    
            current_query_end = current_query_start + len(y_query_array[j]) - 1
    
            modified_query = modify_query_librosa(y_query_array[j], sr)
    
            mfcc_query = do_feature_extraction(modified_query, sr)
    
            start_index, end_index, min_dist = ld.least_distance_mfcc(mfcc_reference.T, mfcc_query.T)
    
            if (
                abs((current_query_start / sr) - ((start_index * hop_samples) / sr))
                > MATCH_THRESHOLD
                or abs((current_query_end / sr) - ((end_index * hop_samples) / sr))
                > MATCH_THRESHOLD
            ):
                print("*** MISMATCH ***")
                mismatches += 1
    
            print(
                "i %d j %d. Expected start %d end %d. Returned start %d end sec %d"
                % (
                    i,
                    j,
                    current_query_start,
                    current_query_end,
                    (start_index * hop_samples),
                    (end_index * hop_samples),
                )
            )
    
        assert current_query_end == len(y_reference) - 1
    
        queries_left = queries_left - current_number_of_queries_in_reference
    
    assert queries_left == 0
    assert query_file_number == query_file_count
    
    print("DONE!. Mismatches: ", mismatches)

if __name__ == '__main__':
    main()
