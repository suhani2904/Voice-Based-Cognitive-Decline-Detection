from transcript import convert_audio_to_text
import spacy 
import re
from itertools import tee
import librosa
import numpy as np

def analyze_false_starts(sentences , results):

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    
    for prev_sent, curr_sent in pairwise(sentences):
        prev_words = prev_sent.split()[-2:]
        curr_words = curr_sent.split()[:2]
        prev_sentence = ' '.join(prev_words).replace('.' , '').replace(',' , '')
        curr_sentence = ' '.join(curr_words).replace('.' , '').replace(',' , '')

        if len(prev_words) > 0 and len(curr_words) > 0:
            if prev_sentence == curr_sentence:
                results['false_starts'] += 1

def analyze_hedge_words_and_hestitations(transcript , results):
    sentences= transcript.lower()
    hedge_words = {"sort of", "kind of", "you know", "i mean", "maybe", "perhaps"}
    hesitation_markers = {"uh", "um", "er", "ah", "uhm"}

    def count_phrases(text, phrases):
        return sum(len(re.findall(rf'\b{re.escape(phrase)}\b', text)) for phrase in phrases)
    
    results["hedge_words"] = count_phrases(sentences, hedge_words)
    results["hesitations"] = count_phrases(sentences, hesitation_markers)




def detect_silent_pauses(audio_path, results, min_silence_duration=300, top_db=40 ):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    non_silent_intervals = librosa.effects.split(
        y, 
        top_db=top_db,
        frame_length=2048,
        hop_length=512
    )
    

    for i in range(1, len(non_silent_intervals)):
        prev_end = non_silent_intervals[i-1][1]
        curr_start = non_silent_intervals[i][0]
        pause_duration = (curr_start - prev_end) / sr * 1000  
        if pause_duration >= min_silence_duration:
            results["silent_pauses"] += 1
    

def detect_incomplete_sentence(transcript , results , nlp):
    incomplete_markers = {"...", "uh,", "um,", "and then"}

    for sent in transcript.split('.'):
        sent = sent.strip().lower()

        if not sent:
            continue

        for marker in incomplete_markers:
            results["incomplete_sent"] += sent.endswith(marker)
        if not any(token.pos_ == "VERB" for token in nlp(sent)):
            results["incomplete_sent"] += 1


def calculate_pitch_variability(audio_path , results):
    y, sr = librosa.load(audio_path)
    f0 = librosa.yin(y, fmin=60, fmax=300)  
    f0_clean = f0[f0 > 0]  
    results["pitch_variability"] = np.std(f0_clean)

def calculate_speech_rate(words , results):
    duration_sec = words[-1].end - words[0].start

    results["speech_rate"] = len(words)/(duration_sec/60)

def naming_words_association(sentences , results):

    vague_terms = {"thing", "stuff", "something"}

    for sent in sentences:
        words = sent.split()

        for terms in vague_terms:
            results["naming_association"] += sent.count(terms)

        results["word_repetitions"] += sum(words[i] == words[i+1] for i in range(len(words)-1))

def extract_features(audio_path):
    transcript , words = convert_audio_to_text(audio_path)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcript)
    sentences = [sent.text.strip().lower() for sent in doc.sents]
    results = {
    "false_starts" : 0,
    "hedge_words" : 0,
    "hesitations": 0,
    "silent_pauses" :0,
    "incomplete_sent" : 0,
    "pitch_variability" : 0,
    "speech_rate" : 0,
    "naming_association":0,
    "word_repetitions" :0 , 
    "hesitations/length" : 0
}

    analyze_false_starts(sentences , results)
    analyze_hedge_words_and_hestitations(transcript , results)
    detect_silent_pauses(audio_path , results)
    detect_incomplete_sentence(transcript , results , nlp)
    calculate_pitch_variability(audio_path , results)
    calculate_speech_rate(words , results)
    naming_words_association(sentences , results)
    results["hesitations/length"] = results["hesitations"] / len(transcript)

    return transcript , results




            






    



