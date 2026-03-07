import os
import numpy as np
import pandas as pd
from Segments import segment_audio
from acoustic_features import acoustic_features
from linguistic_features import transcribe_and_embed

def build_dataset(base_dir, label):
    rows = []
    for person_folder in os.listdir(base_dir):
        person_path = os.path.join(base_dir, person_folder)
        if not os.path.isdir(person_path):
            print("error")
            continue  # skip if not a folder

        for file in os.listdir(person_path):
            
            audio_path = os.path.join(person_path, file)
          
            # Segment the audio into 6-sec chunks
            if audio_path.endswith(".wav"):
                segments = segment_audio(audio_path, segment_len=6, overlap=3)

                for seg in segments:
                    # Extract acoustic + linguistic features
                    acoustic = acoustic_features(seg)  # np.array
                    linguistic = transcribe_and_embed(seg)  # np.array

                  
                    if linguistic is not None and acoustic is not None:
                        
                        # Combine features
                        features = np.concatenate([acoustic, linguistic])

                        # Store features with label
                        rows.append(np.hstack([features, label]))
                    else:
                        print("no_features")

    return pd.DataFrame(rows)



dementia_dir = "dementia"
no_dementia_dir = "no_dementia"


df_no_dementia = build_dataset(no_dementia_dir, label=0)
df_dementia = build_dataset(dementia_dir , label = 1)

df = pd.concat([df_dementia, df_no_dementia], ignore_index=True)

df.to_csv("final_features_dataset.csv", index=False)
