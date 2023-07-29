
import enum
import numpy as np
from math import inf

class Direction(enum.Enum):

    DEFAULT = 0 
    DIAGONAL = 1
    RIGHT = 2
    DOWN = 3

def dist(seq1, seq2):
    
    r,c = len(seq1), len(seq2)
    
    distances = np.zeros((r, c))
    
    print("r and c :", r,c)
    
    for i in range(0, len(seq1)):
        for j in range(0, len(seq2)):
            distances[i][j] = abs(seq1[i] - seq2[j])
    
    return distances

def nw_int(seq1, seq2):
    
    # Calculate pairwise distances between MFCC vectors
    distances = dist(seq1, seq2)
    
    print(distances)
    
    # Initialize matrix for accumulated distances
    acc_dist = np.zeros(distances.shape)
    direction_matrix = np.zeros(distances.shape)
    
    # the first mfcc should be that of the larger query, the second the smaller reference
    assert acc_dist.shape[0] > acc_dist.shape[1]
    
    #direction_matrix = np.array(distances.shape)
    #direction_matrix.fill(Direction.DEFAULT.value)
    """
    This is the change I made today. Yesterday, I had left the first row and column as is.
    Here, I am accumulating the first row and column.
    Yesterday I left as is because I did not want to introduce any penalty for an element,say,
    6th in a series of 10, that could be the start of the match.
    But realised that even with accumulating the distances, 6th element would not be at a disavantage
    """
    acc_dist[0, 0] = distances[0, 0]
    for i in range(1, acc_dist.shape[0]):
        acc_dist[i, 0] = distances[i, 0] + acc_dist[i-1, 0]
        print("R acc dist i : d : ", i, acc_dist[i, 0])
    for i in range(1, acc_dist.shape[1]):
        acc_dist[0, i] = distances[0, i] + acc_dist[0, i-1]
        print("C acc dist i : d : ", i, acc_dist[0, i])
    
    print("ACC DIST :", acc_dist)
    
    for i in range(1, acc_dist.shape[0]):
        
        for j in range(1, acc_dist.shape[1]):
            print("i j :", i, j)
            curr_dist = []
            curr_dir = []
            for d in Direction:
                if (d.name == 'DIAGONAL'):
                    print("DIA : ", acc_dist[i-1, j-1])
                    curr_dist.append(acc_dist[i-1, j-1])
                    curr_dir.append(Direction.DIAGONAL.value)
                if (d.name == 'RIGHT'):
                    print("RIT : ", acc_dist[i, j-1])
                    curr_dist.append(acc_dist[i, j-1])
                    curr_dir.append(Direction.RIGHT.value)
                if (d.name == 'DOWN'):
                    print("DWN : ", acc_dist[i-1, j])
                    curr_dist.append(acc_dist[i-1, j])
                    curr_dir.append(Direction.DOWN.value)
            
            assert len(curr_dist) == 3
            min_dist = inf
            for k in range(0, len(curr_dist)):
                if (curr_dist[k] < min_dist):
                    min_dist = curr_dist[k]
                    direction_matrix[i][j] = curr_dir[k]
                    print("k min_dist dir", k, min_dist, curr_dir[k])
            
            assert direction_matrix[i,j] != Direction.DEFAULT.value
            assert min_dist != inf
            print("min dist : dir :", min_dist, direction_matrix[i,j])
            acc_dist[i, j] = distances[i, j] + min_dist

    i = acc_dist.shape[0] - 1
    j = acc_dist.shape[1] - 1
    
    q_end_idx = -1
    max_loop = acc_dist.shape[0] + 1
    
    print("END i j :", i, j)
    print(acc_dist)
    print(direction_matrix)
    
    while j > 0:
        print("DIR :", direction_matrix[i][j])
        if (direction_matrix[i][j] == Direction.DIAGONAL.value):
            print("DIA . i and end :", i, q_end_idx)
            if (q_end_idx == -1):
                q_end_idx = i
            i = i - 1
            j = j - 1
            
        elif (direction_matrix[i][j] == Direction.RIGHT.value):
            print("RIGHT")
            j = j - 1
        elif (direction_matrix[i][j] == Direction.DOWN.value):
            print("DOWN")
            i = i - 1
        else:
            assert 0
        max_loop = max_loop - 1
        #print("max loop :", max_loop)
        assert max_loop > 0

    assert i >= 0
    assert q_end_idx != -1
    
    q_start_idx = i

    return q_start_idx, q_end_idx

seq1 = [8,4,5,9,10,2,0,4]
seq2 = [9,10]
start,end = nw_int(seq1, seq2)
print("start = %d end = %d" %(start, end))
