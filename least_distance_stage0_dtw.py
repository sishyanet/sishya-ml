# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:42:46 2023

@author: Raghavan

This is a code to test the least distance algorithm.

The purpose of the algorithm is to identify the best match for a subset
within a superset of mfcc's.

This code will do the following - 

1. Take an audio file, divide it into segments of REFERENCE_DURATION secs.

2. Calculate the mfcc's for each segment.

(If the reference was of length 5 secs, the number of samples would be
5 * sample rate. If the sample rate was 22K this would mean 110K samples.
For 110K samples, there would be 216 mfcc's sets with default librosa values.)

3. Create NUM_QUERIES_IN_REFERENCE number of queries within each reference.

4. With the query formed, the code will try to find the area within the
reference, where the query would match best. This is done with the least
distance algorithm.

5.The algorithm starts from each index in the reference (which in our
example runs from 0 to 215) and calculates the cost of aligning the query
(of size 53 in our example) and the end of the alignment within the reference.

6. The start reference with the least distance is taken as the
begining of the query within the reference and the end of the least
distance alignment is taken as the end of the query within the reference.

7. The start and end returned by the algorithm is compared against the
actual start and end.

"""

import librosa
import numpy as np
from scipy.spatial.distance import cdist
from math import inf
from typing import Tuple



"""
least_distance_mfcc_dtw

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


def main():
    file = r"data/ds1.wav"
    y, sr = librosa.load(file)

    REFERENCE_DURATION = 5  # in secs

    NUM_QUERIES_IN_REFERENCE = 125

    # number of samples in the reference duration
    reference_samples = REFERENCE_DURATION * sr

    n_errors, n_correct = 0, 0

    # doing len(y) - reference_samples to leave out the last part, which maybe small enough to be lesser than n_fft
    for i in range(0, len(y) - reference_samples, reference_samples):

        y_reference = y[i : i + reference_samples]

        mfcc_reference = librosa.feature.mfcc(y=y_reference, sr=sr, center=False)
        """
        librosa returns an array with n_mfcc (number of mfcc's per frame)
        number of columns. having an array with the number of columns equal to the number
        of mfcc sets (number of frames) and number of rows equal to n_mfcc
        is more easy to visualise and more importantly cdist function which
        calculates the distance matrix requires that the number of rows to be
        the same for the two arrays. Hence taking a traspose.
        """
        mfcc_reference = mfcc_reference.T

        num_mfccs_in_query = mfcc_reference.shape[0] / NUM_QUERIES_IN_REFERENCE

        # for every reference, NUM_QUERIES_IN_REFERENCE queries are created.
        for j in range(0, NUM_QUERIES_IN_REFERENCE):

            # typecasting as int since query_start will be used as array index
            # and only an integer can be an array index
            query_start = int(j * num_mfccs_in_query)
            query_end = query_start + int(num_mfccs_in_query)

            mfcc_query = mfcc_reference[query_start:query_end]

            # find the best match for the query within the reference
            start_index, end_index, min_distance = least_distance_mfcc_dtw(
                mfcc_reference, mfcc_query
            )

            # expect the best match to be the query itself
            if (
                start_index != query_start
                or end_index != query_end - 1
                or min_distance != 0
            ):

                print(
                    "i %d j %d. exp start %d [got %d] end %d [got %d]. \
                      Query length %d min distance %d"
                    % (
                        i,
                        j,
                        query_start,
                        start_index,
                        query_end - 1,
                        end_index,
                        mfcc_query.shape[0],
                        min_distance,
                    )
                )
                n_errors += 1
            else:
                n_correct += 1

    print(f"DONE : correct {n_correct}, error {n_errors}")


if __name__ == '__main__':
    main()
