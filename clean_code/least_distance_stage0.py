# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:19:51 2024

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

import least_distance as ld
import librosa

check_with_dtw = 0

def main():
    file = r"C:\Users\Lenovo\Desktop\dataset\ds_4min\ds1_4.wav"
    y, sr = librosa.load(file)

    REFERENCE_DURATION = 5  # in secs

    NUM_QUERIES_IN_REFERENCE = 4

    # number of samples in the reference duration
    reference_samples = REFERENCE_DURATION * sr

    # doing len(y) - reference_samples to leave out the last part, which maybe small enough to be lesser than n_fft
    for i in range(0, len(y) - reference_samples, reference_samples):

        y_reference = y[i : i + reference_samples]

        mfcc_reference = librosa.feature.mfcc(y=y_reference, sr=sr, center=False)
        mfcc_reference = mfcc_reference.T

        num_mfccs_in_query = mfcc_reference.shape[0] / NUM_QUERIES_IN_REFERENCE

        # for every reference, NUM_QUERIES_IN_REFERENCE queries are created.
        for j in range(0, NUM_QUERIES_IN_REFERENCE):

            # typecasting as int since query_start will be used as array index
            # and only an integer can be an array index
            query_start = int(j * num_mfccs_in_query)
            query_end = query_start + int(num_mfccs_in_query)

            mfcc_query = mfcc_reference[query_start:query_end]

            if (check_with_dtw == 1):
                # find the best match for the query within the reference
                start_index, end_index, min_distance = ld.least_distance_mfcc_dtw(
                    mfcc_reference, mfcc_query
                )
            else:
                # find the best match for the query within the reference
                start_index, end_index, min_distance = ld.least_distance_mfcc(
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

    print("DONE :")


if __name__ == '__main__':
    main()

