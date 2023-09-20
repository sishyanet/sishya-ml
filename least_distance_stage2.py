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
from scipy.spatial.distance import cdist
from math import inf
from typing import List, Tuple
import fnmatch
import os
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

NUM_FEATURES: int = 36
hop_samples: int = 2048
window_samples: int = 4096
NUM_QUERIES_IN_REFERENCE: int = 10
LIBROSA_SEMITONE_STEP: float = 0.3
LIBROSA_TIME_STRETCH: float = 1.1
MATCH_THRESHOLD: float = 0.5  # in secs
PITCH_SHIFT: int = 1
TIME_STRETCH: int = 0
DO_MFCC_FEATURE: int = 1
query_directory: str = (
    r"C:\Users\Lenovo\Desktop\dataset\ds_4min\sishyaapp_recordings_wav"
)

"""

get_next_distance
    
parameters - 
reference_idx is the current index in the reference
query_idx is the current query index
distances is the distance matrix

functionality:
1. If the alignment process is currently in the middle of the query
and the reference (that is , when there are not any edge conditions) 
the function will use the standard dtw operations of match,delete or insert
2. If the query end is reached, the variable distance will be set to -1
to indicate end of the alignment process
3. If the reference end is reached, the query elements hereafter will
be aligned with the end element of the reference

reference_idx and query_idx will hold the current locations within the
reference and query when the function begins. (for eg: say 4 and 5 are
the reference_idx and query_idx)
when the next best alignment (least distance among the 3) has been
calculated the reference_idx and query_idx will be set accordingly,
meaning they now hold the locations within the reference and query from
where the alignment process needs to be continued. (if the dtw operation
was a match, then the current reference_idx and query_idx would
be 5 and 6 in our example)

similarly for the edge conditions the reference_idx and query_idx
are set accordingly.

returns - 
1. the next distance which is the minimum of the 3 possible distances
2. the next reference index
3. the next query index

"""

def get_next_distance(
    reference_idx: int, query_idx: int, distances: np.ndarray
) -> Tuple[float, int, int]:
    distance: float = 0

    # Reached the end of the query
    if query_idx == distances.shape[1] - 1:
        distance = -1
    else:
        # at the end of the reference, align all the coming queries to this end
        if reference_idx == distances.shape[0] - 1:
            query_idx = query_idx + 1
            assert query_idx < distances.shape[1]
            distance = distances[reference_idx][query_idx]
        else:
            temp_distance: List[float] = []
            min_dist: float = inf
            min_idx: int = -1

            # match
            temp_distance.append(distances[reference_idx + 1][query_idx + 1])
            # insert
            temp_distance.append(distances[reference_idx][query_idx + 1])
            # delete
            temp_distance.append(distances[reference_idx + 1][query_idx])

            for k in range(0, len(temp_distance)):
                if temp_distance[k] < min_dist:
                    min_dist = temp_distance[k]
                    min_idx = k

            if min_idx == 0:
                reference_idx, query_idx = reference_idx + 1, query_idx + 1
            elif min_idx == 1:
                reference_idx, query_idx = reference_idx, query_idx + 1
            else:
                assert min_idx == 2
                reference_idx, query_idx = reference_idx + 1, query_idx

            distance = min_dist

    return distance, reference_idx, query_idx


"""
calculate_min_distance

parameters:
1. reference_idx : the current start of the reference. 
runs from 0 to the number of elements in the reference - 1
2. distances : the distance matrix
3. min_distance : the minimum distance till now. 

functionality:

The reference_idx is aligned with the start of the query.
Then the dtw algorithm is used to find the next alignments.
The current query element could be aligned with the next reference (insert)
The current reference could be aligned with the next query (delete)
The next reference and next query could be aligned with each other (match)

If the current match iteration exceeds the current minimum distance, the
function quits. Since the purpose is to find the match with the minimum
distance, no point in doing further alignment when the distance is not
going to be the best one.

returns:
1. total_distance : the distance for aligning query with the reference
begining at the reference_idx.
2. reference_end_idx : the end of alignment of the query 
within the reference

"""

def calculate_min_distance(
    reference_idx: int, distances: np.ndarray, min_distance: float
) -> Tuple[float, int]:
    total_distance: float = 0
    query_idx: int = 0

    next_distance: float = distances[reference_idx][0]

    while next_distance >= 0:

        total_distance = total_distance + next_distance
        next_distance, reference_idx, query_idx = get_next_distance(
            reference_idx, query_idx, distances
        )

        # no need to traverse any more, this is not the best match
        if total_distance > min_distance:
            break

    return total_distance, reference_idx


"""
least_distance_mfcc

parameters:
1. mfcc_reference : the reference mfcc array
2. mfcc_query : the query mfcc array

functionality:
with every element in the reference as the start the function
calculates the distance of aligning the query with the reference
begining at the current element. for eg. the function first finds the
distance of algning the query with the reference begining at element 0.
if the query is of length 10, it could be alinged with elements 0-8 of
the reference with a total distance of 4.5. Then in the next iteration
with the start of the reference being the first element, the 10 element 
query could be aligned with 1-11 with a total distance of 6.7. Then with
2-10, 3-11 etc. with the respective distances. The best match is the one
with the least distance.

returns:
1. the start of the best match
2. end of the best match
"""


def least_distance_mfcc(
    mfcc_reference: np.ndarray, mfcc_query: np.ndarray
) -> Tuple[int, int]:
    min_distance: float = inf
    start_index: int = 0
    end_index: int = 0

    distances: np.ndarray = cdist(mfcc_reference.T, mfcc_query.T, "euclidean")

    for i in range(0, distances.shape[0]):
        current_distance, end = calculate_min_distance(i, distances, min_distance)
        if current_distance < min_distance:
            min_distance = current_distance
            start_index = i
            end_index = end

    return start_index, end_index

"""
modify_query_audiomentations:

Not used right now. Query modifications are done with inbuilt librosa
functions. Librosa modifications are a constant. 
Audiomentation modifications work on a range of random parameters and
hence will not give the same modified output every run.

"""

def modify_query_audiomentations(y_query: np.ndarray, sr: int) -> np.ndarray:

    augment = Compose(
        [
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-1, max_semitones=1, p=1)
            # Shift(p=0.5),
        ]
    )

    y_query = augment(samples=y_query, sample_rate=sr)

    return y_query


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

#The directory holds segmented sishya app files. The files will be
#concatenated to form the reference.
query_file_count: int = len(fnmatch.filter(os.listdir(query_directory), "*.*"))
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

        f = r"%s\%d.wav" % (query_directory, query_file_number)
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

        start_index, end_index = least_distance_mfcc(mfcc_reference, mfcc_query)

        if (
            abs((current_query_start / sr) - ((start_index * hop_samples) / sr))
            > MATCH_THRESHOLD
            or abs((current_query_end / sr) - ((end_index * hop_samples) / sr))
            > MATCH_THRESHOLD
        ):
            print("*** MISMATCH ***")
            mismatches += 1

        print(
            "i %d j %d. Expected start %f end %f. Returned start %f end sec %f"
            % (
                i,
                j,
                current_query_start / sr,
                current_query_end / sr,
                (start_index * hop_samples) / sr,
                (end_index * hop_samples) / sr,
            )
        )

    assert current_query_end == len(y_reference) - 1

    queries_left = queries_left - current_number_of_queries_in_reference

assert queries_left == 0
assert query_file_number == query_file_count

print("DONE!. Mismatches: ", mismatches)
