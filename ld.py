
import numpy as np
from math import inf

def get_next_dist(idx, i, seq1, seq2):

    distance = 0

    if (i == len(seq2) - 1):
        distance = -1
    else:
        #at the end of the reference, align all the coming queries to this end
        if (idx == len(seq1) - 1):
            i = i + 1
            assert i < len(seq2)
            distance = abs(seq1[idx] - seq2[i])
        else:
            
            curr_dist = []
            min_dist = inf
            min_idx = -1
            
            curr_dist.append(abs(seq1[idx+1] - seq2[i+1]))
            curr_dist.append(abs(seq1[idx] - seq2[i+1]))
            curr_dist.append(abs(seq1[idx+1] - seq2[i]))

            for j in range(0, len(curr_dist)):
                if (curr_dist[j] < min_dist):
                    min_dist = curr_dist[j]
                    min_idx = j

            if (min_idx == 0):
                idx,i = idx+1, i+1
            elif (min_idx == 1):
                idx,i = idx, i+1
            else:
                assert min_idx == 2
                idx,i = idx+1, i
                
            distance = min_dist 
            
    return distance, idx, i 

def calc_min_dist(idx, seq1, seq2, min_dist):

    tot_dist = 0
    i = 0
    
    next_dist = abs(seq1[idx] - seq2[i])
    
    while (next_dist >= 0):
        
        tot_dist = tot_dist + next_dist
        next_dist, idx, i =  get_next_dist(idx, i, seq1, seq2)
        
        #no need to traverse any more, this is not the best match
        if (tot_dist > min_dist):
            break
                    
    return tot_dist, idx

def ld_int(seq1, seq2):

    min_dist = inf
    start_idx, end_idx = 0,0

    for i in range(0, len(seq1)):
        curr_dist, end = calc_min_dist(i, seq1, seq2, min_dist)
        if (curr_dist < min_dist):
            min_dist = curr_dist
            start_idx = i
            end_idx = end

    return start_idx, end_idx

seq1 = [66,32,45,102,44,989,244,1,234,2]
seq2 = [100,50]
start_idx,end_idx = ld_int(seq1, seq2)
print("start = %d end = %d" %(start_idx, end_idx))
