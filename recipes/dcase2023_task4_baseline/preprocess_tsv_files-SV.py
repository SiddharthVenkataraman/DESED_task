import pandas as pd
import os
import yaml
import glob
import numpy as np

# Script to handle problems with incomplete DESED input data


def remove_missing_files_from_tsv(path_to_tsv, path_to_data_folder):
    """ Load .tsv file found at <path_to_tsv>, and remove rows that point to a file that isn't found in <path_to_data_folder>"""

    # Load DF
    DF = pd.read_csv(path_to_tsv, sep="\t")

    # Load filenames from DF, and from folder
    files_in_DF = DF['filename'].unique()
    files_in_folder_abs_path = glob.glob(f'{path_to_data_folder}{os.sep}*.wav')
    files_in_folder = [x.split(os.sep)[-1] for x in files_in_folder_abs_path]

    # Find files that are in _DF, but not in _folder
    missing_files = list(set(list(files_in_DF)).difference(files_in_folder))

    # Remove rows in DF that have filename found in <missing_files>
    DF['remove'] = 0
    for missing_file in missing_files:
        DF.loc[DF['filename'] == missing_file, 'remove'] = 1
    DF_new = DF.loc[DF['remove'] == 0].copy()
    DF_new.drop(columns=['remove'], inplace=True)


    # Save the new DF in same location as the old, with a "_removed_missing_SV" suffix
    tsv_filename_old = path_to_tsv.split(os.sep)[-1].split('.')[0]
    tsv_filename_new = f'{tsv_filename_old}_removed_missing_SV'
    path_to_tsv_new = f'{os.path.dirname(path_to_tsv)}{os.sep}{tsv_filename_new}.tsv'

    DF_new.to_csv(path_to_tsv_new, '\t', index=False)

    print(f'Processed file "{tsv_filename_old}"\n'
          f'Total files expected = {len(files_in_DF)}\n'
          f'Files found = {len(files_in_folder)}\n'
          f'Percentage missing = {100*len(missing_files)/len(files_in_DF):.1f}% \n\n')


if __name__ == '__main__':

    path_to_conf_file = '/home/siddharth/PycharmProjects/DESED_task/recipes/dcase2023_task4_baseline/confs/default.yaml'
    with open(path_to_conf_file, "r") as f:
        config = yaml.safe_load(f)

    path_to_tsvs_and_data_folders = [
        (config["data"]["synth_tsv"], config["data"]["synth_folder_44k"]),
        (config["data"]["strong_tsv"], config["data"]["strong_folder_44k"]),
        (config["data"]["weak_tsv"], config["data"]["weak_folder_44k"]),
        (config["data"]["synth_val_tsv"], config["data"]["synth_val_folder_44k"]),
        (config["data"]["synth_val_dur"], config["data"]["synth_val_folder_44k"]),
        (config["data"]["test_tsv"], config["data"]["test_folder_44k"]),
        # (config["data"]["test_dur"], config["data"]["test_folder_44k"]),
    ]

    for (path_to_tsv, path_to_data_folder) in path_to_tsvs_and_data_folders:
        remove_missing_files_from_tsv(path_to_tsv, path_to_data_folder)
