# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:02:39 2024

@author: Raghavan
"""

import numpy as np
import librosa
from scipy.spatial.distance import cdist
from math import inf
from typing import Tuple

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

    distance = 0

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

            temp_distance = []
            min_dist = inf
            min_idx = -1

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

    total_distance = 0
    query_idx = 0

    next_distance = distances[reference_idx][0]

    reference_end_idx = reference_idx

    while next_distance >= 0:

        total_distance = total_distance + next_distance
        next_distance, reference_end_idx, query_idx = get_next_distance(
            reference_end_idx, query_idx, distances
        )

        # no need to traverse any more, this is not the best match
        if total_distance > min_distance:
            break

    return total_distance, reference_end_idx


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
3. the distance of the best match
"""


def least_distance_mfcc(
    mfcc_reference: np.ndarray, mfcc_query: np.ndarray
) -> Tuple[int, int, float]:

    min_distance = inf
    start_index, end_index = 0, 0

    distances = cdist(mfcc_reference, mfcc_query, "euclidean")

    for i in range(0, distances.shape[0]):
        current_distance, end = calculate_min_distance(i, distances, min_distance)
        if current_distance < min_distance:
            min_distance = current_distance
            start_index = i
            end_index = end

    return start_index, end_index, min_distance


def least_distance_mfcc_dtw(
    mfcc_reference: np.ndarray, mfcc_query: np.ndarray
) -> Tuple[int, int, float]:

    min_distance = inf
    start_index, end_index = 0, 0

    D, wp = librosa.sequence.dtw(mfcc_reference.T, mfcc_query.T, metric='euclidean', subseq=True)
    end_index = np.argmin(D[-1, :])
    min_distance = D[-1, end_index]
    start_index = end_index - mfcc_query.shape[0] + 1

    return start_index, end_index, min_distance

