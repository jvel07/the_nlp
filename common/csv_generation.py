import glob
import os

import pandas as pd
from tqdm import tqdm


def create_csv_compare_23(in_path, out_file):
    """Function to create csv file for the Sclerosis Multiple Corpus with labels of the form:
    'file_name', 'label'

    :param in_path: string, name of the output file preceded with a desired parent directory. E.g.: a/path/train_set
    :param out_file: string, path to the dataset containing the utterances.
    :return: List, final csv list.
    """

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # reading directories
    for folder_name in ['train', 'dev', 'test']:
        audio_list = glob.glob('{0}/{1}/*.wav'.format(in_path, folder_name))
        audio_list.sort()

        final_list = []
        # audio_list = audio_list[0:5]
        for wav_path in tqdm(audio_list, total=len(audio_list)):
            file_name = os.path.basename(wav_path)
            final_list.append(wav_path + ',' + str(file_name))

        df = pd.DataFrame([sub.split(",") for sub in final_list], columns=['file', 'name'])

        final_name = os.path.join(out_file, folder_name + '.csv')
        df.to_csv(final_name, sep=',', index=False)
        print("Data saved to {}".format(final_name))
