from datasets import load_dataset, DownloadMode
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import sys
sys.path.append('../../')

from common.utils import load_config
from common.utils_preprocess import PreprocessFunction, get_transcription
from common.csv_generation import create_csv_compare_23

# Loading configuration
# config = utils.load_config('config/config_bea16k.yml')  # provide the task's yml
config = load_config('../../conf/config_requests.yml')
model_name = config['pretrained_model_details']['checkpoint_path']
task = config['task']  # name of the dataset
audio_path = config['paths']['audio_path']  # path to the audio files of the task
label_path = config['paths']['to_labels']  # path to the labels of the dataset
save_path = config['paths']['to_save_metadata']  # path to save the csv file containing info of the dataset (metadata)


create_csv_compare_23(in_path=audio_path, out_file=label_path)  #

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None  # for English to English
forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe") # for French to French

# Load data in HF 'datasets' class format
data_files = {
    "train": label_path + 'train.csv',  # this is the metadata
    "dev": label_path + 'dev.csv',  # this is the metadata
    "test": label_path + 'test.csv'  # this is the metadata
}


dataset = load_dataset("csv", data_files=data_files, delimiter=",", cache_dir=config['hf_cache_dir'],
                       download_mode=DownloadMode['REUSE_DATASET_IF_EXISTS'])

for data_set in ['train', 'dev', 'test']:
    sub_set = dataset[data_set]
    preprocess = PreprocessFunction(processor, sampling_rate=16000)
    print("Preprocessing {} dataset".format(data_set))
    dataset_loader = sub_set.map(preprocess.preprocess_function,
                                      batched=True,
                                      batch_size=1,
                                      keep_in_memory=False
                                      )
    print("Generating transcriptions for {} dataset".format(data_set))
    transcripts_df = get_transcription(dataset_loader, processor, model, forced_decoder_ids)

    transcripts_df.to_csv('{0}/{1}_transcriptions.csv'.format(save_path, data_set), index=False)

