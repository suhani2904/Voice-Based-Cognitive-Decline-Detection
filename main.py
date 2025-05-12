from extract_features import extract_features
import os
import pandas as pd

dementia_audio_path = "Audio_clips_dementia"
nodementia_audio_path = "Audio_clips_nodementia"

def create_dataset(dementia_audio_path , nodementia_audio_path):
    data = []

    for root, _, files in os.walk(dementia_audio_path):
        for file in files:
            features = extract_features(os.path.join(root, file))
            if features:
                features['label'] = 1
                data.append(features)

    for root, _, files in os.walk(nodementia_audio_path):
        for file in files:
            features = extract_features(os.path.join(root, file))
            if features:
                features['label'] = 0
                data.append(features)

    return pd.DataFrame(data)

df = create_dataset(dementia_audio_path , nodementia_audio_path)
df.to_csv("output.csv" , index = False)
