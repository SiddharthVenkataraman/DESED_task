import os
import shutil
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def split_df(df: pd.DataFrame, dur_df: pd.DataFrame, split_ratios: dict, base_folder: str, random_state: int = 42):
    """
    Splits a dataframe into sets and moves files into corresponding folders.
    Args:
        df (pd.DataFrame): Dataframe to split.
        dur_df (pd.DataFrame): Dataframe with the duration of each file.
        split_ratios (dict): Ratios to split the dataframe.
        base_folder (str): Base folder where the original files are located.
        random_state (int): Random state for the split. Defaults to 42.
    Returns:
        dict: Dictionary of tuples (dataframe, duration dataframe) for each set.
    """
    # Initial split to separate Test Set
    remaining, test = train_test_split(df, test_size=split_ratios['test'], random_state=random_state)

    # Further splitting for other sets
    remaining_ratio = 1 - split_ratios['test']
    strong_ratio = split_ratios['strong'] / remaining_ratio
    synth_ratio = (split_ratios['synth_train'] + split_ratios['synth_val']) / remaining_ratio
    weak_ratio = split_ratios['weak'] / remaining_ratio

    strong, remaining = train_test_split(remaining, test_size=1-strong_ratio, random_state=random_state)
    synth, remaining = train_test_split(remaining, test_size=1-synth_ratio/(1-strong_ratio), random_state=random_state)
    weak, unlabeled = train_test_split(remaining, test_size=1-weak_ratio/(1-synth_ratio/(1-strong_ratio)), random_state=random_state)

    # Splitting Synth into Synth Train and Synth Val
    synth_train_ratio = split_ratios['synth_train'] / (split_ratios['synth_train'] + split_ratios['synth_val'])
    synth_train, synth_val = train_test_split(synth, test_size=1-synth_train_ratio, random_state=random_state)

    # Function to update paths and move files
    def update_paths_and_move_files(df, set_name):
        new_folder = os.path.join(base_folder, set_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        df['filename'] = df['filename'].apply(lambda x: os.path.join(set_name, os.path.basename(x)))
        for file in df['filename']:
            src = os.path.join(base_folder, os.path.basename(file))
            dst = os.path.join(base_folder, file)
            shutil.move(src, dst)

    # Update paths and move files for each set
    for set_name, set_df in zip(['strong', 'synth_train', 'synth_val', 'weak', 'unlabeled', 'test'], [strong, synth_train, synth_val, weak, unlabeled, test]):
        update_paths_and_move_files(set_df, set_name)
        
    # Creating duration dataframes for each set
    strong_dur = dur_df[dur_df.filename.isin(strong.filename)]
    synth_train_dur = dur_df[dur_df.filename.isin(synth_train.filename)]
    synth_val_dur = dur_df[dur_df.filename.isin(synth_val.filename)]
    weak_dur = dur_df[dur_df.filename.isin(weak.filename)]
    unlabeled_dur = dur_df[dur_df.filename.isin(unlabeled.filename)]
    test_dur = dur_df[dur_df.filename.isin(test.filename)]
    
    return {
        'strong': (strong, strong_dur),
        'synth_train': (synth_train, synth_train_dur),
        'synth_val': (synth_val, synth_val_dur),
        'weak': (weak, weak_dur),
        'unlabeled': (unlabeled, unlabeled_dur),
        'test': (test, test_dur)
    }
# The 'main' function and the argument parser will need to be updated to handle the new split ratio for synth_val.
# In the main function, ensure that the split ratios dictionary includes 'synth_train' and 'synth_val' separately.
def main(args):
    df = pd.read_csv(args.input_tsv, sep="\t")
    dur_df = pd.read_csv(args.input_dur, sep="\t")

    # Split dataframe and move files
    split_ratios = {
        'strong': float(args.split_ratios[0]),
        'synth_train': float(args.split_ratios[1]),
        'synth_val': float(args.split_ratios[2]),
        'weak': float(args.split_ratios[3]),
        'unlabeled': float(args.split_ratios[4]),
        'test': float(args.split_ratios[5])
    }
    split_dfs = split_df(df, dur_df, split_ratios, args.base_folder)

    # Save split dataframes
    for set_name, (data_df, dur_df) in split_dfs.items():
        output_file = f"{set_name}.tsv"
        duration_file = f"{set_name}_duration.tsv"
        data_df.to_csv(os.path.dirname(args.input_tsv) + "/metadata/" + output_file, sep="\t", index=False)
        dur_df.to_csv(os.path.dirname(args.input_tsv) + "/metadata/" + duration_file, sep="\t", index=False)
        print(f"Saved {output_file} and {duration_file}")

if __name__ == '__main__':
    print("Splitting dataframe and moving files")
    parser = argparse.ArgumentParser(description="Split a dataframe into sets and distribute files into corresponding folders.")
    parser.add_argument("--input_tsv", type=str, required=True, help="Path to the input tsv file")
    parser.add_argument("--input_dur", type=str, required=True, help="Path to the input duration tsv file")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder where the original files are located")
    parser.add_argument("--split_ratios", type=float, nargs=6, required=True, help="Ratios to split the dataframe (strong, synth_train, synth_val, weak, unlabeled, test)")
    args = parser.parse_args()
    main(args)
