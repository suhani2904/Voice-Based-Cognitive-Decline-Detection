import librosa 
import numpy as np

def segment_audio(audio_path , segment_len = 6 , overlap = 3):
    y,sr = librosa.load(audio_path , sr = 16000 , mono = True)

    y = librosa.util.normalize(y)
    
    step = segment_len - overlap
    segments = []

    for start in range(0, len(y) , step * sr):
        end = start + segment_len*sr
        if end >len(y):
            pad_length = end - len(y)
            y_seg = np.pad(y[start:], (0, pad_length), mode='constant')

        else:
            y_seg = y[start:end]

        segments.append(y_seg)

    return segments

    