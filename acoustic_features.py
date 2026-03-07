import opensmile 
import librosa 
import numpy as np 
import pandas as pd 
smile = opensmile.Smile( feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals, )

def acoustic_features(segment, sr=16000):
    # reshape for opensmile
    signal = segment.reshape(1, -1)

    # extract opensmile features
    db = smile.process_signal(signal, sr).reset_index(drop=True)

    # pause features from librosa
    intervals = librosa.effects.split(segment, top_db=30)
    total_duration = len(segment) / sr
    speech_durations = [(end - start) / sr for start, end in intervals]
    total_speech_time = np.sum(speech_durations)
    total_pause_time = total_duration - total_speech_time
    pause_ratio = total_pause_time / total_duration if total_duration > 0 else 0
    mean_pause_duration = np.mean(np.diff(intervals[:, 0] / sr)) if len(intervals) > 1 else total_pause_time

    f_dict = {
        "total_pause_time": total_pause_time,
        "pause_ratio": pause_ratio,
        "mean_pause_duration": mean_pause_duration,
    }

    features_1 = pd.DataFrame([f_dict])
    df = pd.concat([features_1, db], axis=1)
    return df.to_numpy().flatten()
