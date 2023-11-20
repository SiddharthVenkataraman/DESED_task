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
    proportions = {key: val / (1 - split_ratios['test']) for key, val in split_ratios.items() if key != 'test'}
    strong, remaining = train_test_split(remaining, test_size=1-proportions['strong'], random_state=random_state)
    synth, remaining = train_test_split(remaining, test_size=1-proportions['synth']/(1-proportions['strong']), random_state=random_state)
    weak, unlabeled = train_test_split(remaining, test_size=1-proportions['weak']/(1-proportions['synth']/(1-proportions['strong'])), random_state=random_state)

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
    for set_name, set_df in zip(['strong', 'synth', 'weak', 'unlabeled', 'test'], [strong, synth, weak, unlabeled, test]):
        update_paths_and_move_files(set_df, set_name)

    # Creating duration dataframes for each set
    strong_dur = dur_df[dur_df.filename.isin(strong.filename)]
    synth_dur = dur_df[dur_df.filename.isin(synth.filename)]
    weak_dur = dur_df[dur_df.filename.isin(weak.filename)]
    unlabeled_dur = dur_df[dur_df.filename.isin(unlabeled.filename)]
    test_dur = dur_df[dur_df.filename.isin(test.filename)]

    return {
        'strong': (strong, strong_dur),
        'synth': (synth, synth_dur),
        'weak': (weak, weak_dur),
        'unlabeled': (unlabeled, unlabeled_dur),
        'test': (test, test_dur)
    }

def main(args):
    df = pd.read_csv(args.input_tsv, sep="\t")
    dur_df = pd.read_csv(args.input_dur, sep="\t")

    # Split dataframe and move files
    split_ratios = {
        'strong': float(args.split_ratios[0]),
        'synth': float(args.split_ratios[1]),
        'weak': float(args.split_ratios[2]),
        'unlabeled': float(args.split_ratios[3]),
        'test': float(args.split_ratios[4])
    }
    split_dfs = split_df(df, dur_df, split_ratios, args.base_folder)

    # Save split dataframes
    for set_name, (data_df, dur_df) in split_dfs.items():
        output_file = f"{set_name}.tsv"
        duration_file = f"{set_name}_duration.tsv"
        data_df.to_csv(output_file, sep="\t", index=False)
        dur_df.to_csv(duration_file, sep="\t", index=False)
        print(f"Saved {output_file} and {duration_file}")

if __name__ == '__main__':
    print("Splitting dataframe and moving files")
    parser = argparse.ArgumentParser(description="Split a dataframe into sets and distribute files into corresponding folders.")
    parser.add_argument("--input_tsv", type=str, required=True, help="Path to the input tsv file")
    parser.add_argument("--input_dur", type=str, required=True, help="Path to the input duration tsv file")
    parser.add_argument("--base_folder", type=str, required=True, help="Base folder where the original files are located")
    parser.add_argument("--split_ratios", type=float, nargs=5, required=True, help="Ratios to split the dataframe (strong, synth, weak, unlabeled, test)")
    args = parser.parse_args()
    main(args)
