
import librosa
from scipy.spatial.distance import cdist

NUM_MFCC = 13
NUM_MELS = 40
SAMPLE_RATE = 16000
FFT_ORDER = 2048
LOWER_FREQ = 133.333300
HIGHEST_FREQ = 6855.497600
WINDOW_LENGTH = int(0.10 * SAMPLE_RATE)
HOP_LENGTH = int(0.04 * SAMPLE_RATE)

def mfccs_to_secs(params, val):
    return val * params.hop_length / params.sr

def secs_to_mfccs(params, val):
    return int(val * params.sr / params.hop_length)

def calc_alignment_deviation(gp):

    total_st_diff, total_end_diff = 0,0
    total_st_diff_mfcc, total_end_diff_mfcc = 0,0
    
    query_labels = gp.query_audio.labels
    params = gp.params
    aligned_indices = gp.aligned_indices
    aligned_indices_mfcc = gp.aligned_indices_mfcc
    
    assert len(query_labels) == len(aligned_indices)
    
    for i in range(0, len(query_labels)):
        exp_st = query_labels[i]["start"]
        exp_end = query_labels[i]["end"]
        act_st = aligned_indices[i]["start"]
        act_end = aligned_indices[i]["end"]
        
        exp_st_mfcc = secs_to_mfccs(params, query_labels[i]["start"])
        exp_end_mfcc = secs_to_mfccs(params, query_labels[i]["end"])
        act_st_mfcc = aligned_indices_mfcc[i]["start"]
        act_end_mfcc = aligned_indices_mfcc[i]["end"]
        
        total_st_diff += abs(exp_st - act_st)
        total_end_diff += abs(exp_end - act_end)

        total_st_diff_mfcc += abs(exp_st_mfcc - act_st_mfcc)
        total_end_diff_mfcc += abs(exp_end_mfcc - act_end_mfcc)

    return total_st_diff, total_end_diff, total_st_diff_mfcc, total_end_diff_mfcc

def construct_labels(params, labelname):
    begin, end = -1,-1
    labels = []
    word="NULL"
    with open(labelname, "r", encoding="utf-8") as fp:
        lines = fp.readlines()
        line_idx = 0
        for line in lines:
            word = 'begin'
            if line.find(word) != -1:
                parts = line.split(": \"")
                assert(end == -1)
                assert(begin == -1)            
                begin = float(parts[1][0:-3])
            word = 'end'
            if line.find(word) != -1:
                parts = line.split(": \"")
                assert(end == -1)
                assert(begin != -1)            
                end = float(parts[1][0:-3])
            word = 'lines'
            if line.find(word) != -1:
                word = lines[line_idx + 1]
                labels.append({"start": begin, 
                               "end": end, "text": word[5:-2]})
                begin, end = -1,-1
            line_idx += 1
    return labels

class GlobalAlignmentPair:

    def __init__(self, params, q_audio, r_audio):

        self.params = params
        self.query_audio = q_audio
        self.ref_audio = r_audio
        
        assert len(q_audio.labels) == len(r_audio.labels)
                
        #distance = cdist(q_audio.mfcc.T, r_audio.mfcc.T)
        #D, self.wp = librosa.sequence.dtw(C = distance, subseq=False)
        _, self.wp = librosa.sequence.dtw(q_audio.mfcc, r_audio.mfcc, backtrack=True)
    
    def _find_alignment_range(self, start, end, wp, search_start):

        alignment_st, alignment_end = -1, -1
        for i in range(search_start, len(wp)):

            if (wp[i][1] == start and alignment_st == -1):
                assert alignment_end == -1
                alignment_st = wp[i][0]
                
            if (wp[i][1] > end):
                assert end == wp[i-1][1]
                alignment_end = wp[i-1][0]
                break

        assert alignment_st != -1
        if (alignment_end == -1):
            alignment_end = wp[i][0]
            
        return alignment_st, alignment_end, i-1

    def get_alignment_indices(self):
        wp = self.wp[::-1]
        ref_labels = self.ref_audio.labels
        self.aligned_indices = []
        self.aligned_indices_mfcc = []
        search_start = 0
        for i in range(0, len(ref_labels)):
            ref_st = secs_to_mfccs(self.params, ref_labels[i]["start"])
            ref_end = secs_to_mfccs(self.params, ref_labels[i]["end"])
            alignment_st, alignment_end, search_start = self._find_alignment_range(ref_st, ref_end, 
                                                             wp, search_start)
            self.aligned_indices_mfcc.append({"start": alignment_st, 
                               "end": alignment_end})
            self.aligned_indices.append({"start": mfccs_to_secs(self.params, alignment_st), 
                               "end": mfccs_to_secs(self.params, alignment_end)})

class AudioText:
    def __init__(self, params, filename, labelname):
        self.filename = filename
        self.labelname = labelname
        audio, sample_rate = librosa.load(filename, sr=params.sr)
        self.labels  = construct_labels(params, labelname)
        self.mfcc = librosa.feature.mfcc(
            y=audio,
            sr=params.sr,
            n_mfcc=params.n_mfcc,
            win_length=params.win_length,
            hop_length=params.hop_length,
            n_mels = params.n_mel,
            n_fft = params.n_fft,
            fmin = params.low_freq,
            fmax = params.high_freq,
            center=False,
        )

class Alignment:

    def __init__(self,
               sr=SAMPLE_RATE, 
               n_fft=FFT_ORDER, 
               hop_length=HOP_LENGTH,
               window_length=WINDOW_LENGTH, 
               n_mfcc=NUM_MFCC, 
               n_mel=NUM_MELS, 
               low_freq=LOWER_FREQ, 
               high_freq=HIGHEST_FREQ):
        self.sr = sr
        self.n_fft = n_fft
        self.win_length = window_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mel = n_mel
        self.low_freq = low_freq
        self.high_freq = high_freq
    
class GlobalAlignment(Alignment):
    pass
