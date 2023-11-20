import os
import shutil
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def split_df(dataframe: pd.DataFrame, dur_df: pd.DataFrame, split_ratios: dict, base_folder: str, random_state: int = 42):
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
    df = dataframe.copy()
    df.filename = df.filename.apply(lambda x: x.split('/')[-1])
    
    # Split dataframe
    print(f"Splitting dataframe with {len(df)} rows")
    weak_size = int(len(df) * split_ratios['weak'])
    unlabeled_size = int(len(df) * split_ratios['unlabeled'])
    test_size = int(len(df) * split_ratios['test'])
    synth_train_size = int(len(df) * split_ratios['synth_train'])
    synth_val_size = int(len(df) * split_ratios['synth_val'])
    
    strong, test = train_test_split(df, test_size=test_size, random_state=random_state)
    strong, weak = train_test_split(strong, test_size=weak_size, random_state=random_state)
    strong, unlabeled = train_test_split(strong, test_size=unlabeled_size, random_state=random_state)
    strong, synth_train = train_test_split(strong, test_size=synth_train_size, random_state=random_state)
    strong, synth_val = train_test_split(strong, test_size=synth_val_size, random_state=random_state)
    
    print(f"Strong: {len(strong)}, Test: {len(test)}, Weak: {len(weak)}, Unlabeled: {len(unlabeled)}, Synth Train: {len(synth_train)}, Synth Val: {len(synth_val)}")
    
    # Creating duration dataframes for each set
    strong_dur = dur_df[dur_df.filename.isin(strong.filename)]
    synth_train_dur = dur_df[dur_df.filename.isin(synth_train.filename)]
    synth_val_dur = dur_df[dur_df.filename.isin(synth_val.filename)]
    weak_dur = dur_df[dur_df.filename.isin(weak.filename)]
    unlabeled_dur = dur_df[dur_df.filename.isin(unlabeled.filename)]
    test_dur = dur_df[dur_df.filename.isin(test.filename)]

    # Function to update paths and move files
    def update_paths_and_move_files(df, set_name):
        new_folder = os.path.join(base_folder, set_name)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        df['filename'] = df['filename'].apply(lambda x: os.path.join(set_name, os.path.basename(x)))
        for file in df['filename']:
            src = os.path.join(base_folder, os.path.basename(file))
            dst = os.path.join(base_folder, file)
            shutil.copy(src, dst)

    # Update paths and move files for each set
    for set_name, set_df in zip(['strong', 'synth_train', 'synth_val', 'weak', 'unlabeled', 'test'], [strong, synth_train, synth_val, weak, unlabeled, test]):
        update_paths_and_move_files(set_df, set_name)
    
    strong_dur['filename'] = strong_dur['filename'].apply(lambda x: os.path.join('strong', os.path.basename(x)))
    synth_train_dur['filename'] = synth_train_dur['filename'].apply(lambda x: os.path.join('synth_train', os.path.basename(x)))
    synth_val_dur['filename'] = synth_val_dur['filename'].apply(lambda x: os.path.join('synth_val', os.path.basename(x)))
    weak_dur['filename'] = weak_dur['filename'].apply(lambda x: os.path.join('weak', os.path.basename(x)))
    unlabeled_dur['filename'] = unlabeled_dur['filename'].apply(lambda x: os.path.join('unlabeled', os.path.basename(x)))
    test_dur['filename'] = test_dur['filename'].apply(lambda x: os.path.join('test', os.path.basename(x)))    
    
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
