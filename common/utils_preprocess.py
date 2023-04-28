import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def speech_to_array(path):
    speech, _ = librosa.load(path, sr=None)
    # batch["speech"] = speech
    return speech

class PreprocessFunction:
    def __init__(self, processor, sampling_rate):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def preprocess_function(self, samples):
        audio_arrays = [speech_to_array(path) for path in samples["file"]]
        inputs = self.processor(audio_arrays, sampling_rate=self.sampling_rate)
        # inputs["speech"] = audio_arrays

        return inputs

def get_transcription(data_loader, processor, model, forced_decoder_ids):
    transcription_df = pd.DataFrame(columns=['wav', 'transcription'])
    for i, data in (pbar := tqdm(enumerate(data_loader, 0), desc="Transcribing", total=len(data_loader))):
        pbar.set_description(f"Transcribing --> {data['file']}")
        input_features = data['input_features']
        input_features = torch.unsqueeze(torch.Tensor(input_features), dim=0)
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        dict_metadata = {
            'wav': data['file'],
            'transcription': transcription
        }
        transcription_df = transcription_df.append(dict_metadata, ignore_index=True)

    return transcription_df

