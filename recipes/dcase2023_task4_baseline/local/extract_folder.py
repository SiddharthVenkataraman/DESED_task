# extract all the audio files in the subfolders of a folder
# and save them in a new folder

import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='extract all the audio files in the subfolders of a folder and save them in a new folder')
parser.add_argument('--input_folder', type=str, default='data/TAU-urban-acoustic-scenes-2020-mobile-development/audio',
                    help='path to the input folder')

parser.add_argument('--output_folder', type=str, default='data/TAU-urban-acoustic-scenes-2020-mobile-development/audio_extracted',
                    help='path to the output folder')

args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith('.wav'):
            shutil.copy(os.path.join(root, file), output_folder)
            
print('done')