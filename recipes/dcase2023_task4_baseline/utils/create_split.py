"""Takes a pandas datadrame and split it according to the split ratio [train, test] (val is the rest)"""

from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def split_df(df: pd.DataFrame, dur_df:pd.DataFrame, split_ratio: list, random_state: int = 42):
    """Splits a dataframe into train, test and val according to the split ratio [train, test] (val is the rest)
    Args:
        df (pd.DataFrame): Dataframe to split
        dur_df (pd.DataFrame): Dataframe with the duration of each file
        split_ratio (list): Ratio to split the dataframe
        random_state (int, optional): Random state for the split. Defaults to 42.
    Returns:
        list: List of dataframes [train, test, val]
    """
    csv = df.copy()
    csv.filename = csv.filename.apply(lambda x: x.split("/")[-1])
    print(f"Total: {len(csv)} files")
    train, temp = train_test_split(csv, train_size=split_ratio[0], random_state=random_state, stratify=df['event_label'])
    val, test = train_test_split(temp, test_size=split_ratio[1], random_state=random_state, stratify=temp['event_label'])
    
    print(f"Train: {len(train)} files, Val: {len(val)} files, Test: {len(test)} files")
    
    train_dur = dur_df[dur_df['filename'].isin(train['filename'])]
    val_dur = dur_df[dur_df['filename'].isin(val['filename'])]
    test_dur = dur_df[dur_df['filename'].isin(test['filename'])]
    
    print(f"Train: {train_dur['duration'].sum():.2f} s, Val: {val_dur['duration'].sum():.2f} s, Test: {test_dur['duration'].sum():.2f} s")
    return [(train, train_dur), (test, test_dur), (val, val_dur)]

def main(args):
    df = pd.read_csv(args.input_tsv, sep="\t")
    dur_df = pd.read_csv(args.input_dur, sep="\t")
    split_dfs = split_df(df, dur_df, args.split_ratio)
    for split_tsv, split_dur_df, output_path in zip(split_dfs, [args.output_train, args.output_test, args.output_val]):
        split_tsv.to_csv(output_path, sep="\t", index=False)
        split_dur_df.to_csv(output_path.replace(".tsv", "_dur.tsv"), sep="\t", index=False)
    
if __name__ == '__main__':
    print("Splitting dataframe")
    parser = argparse.ArgumentParser(description="Split a dataframe into train, test and val according to the split ratio [train, test] (val is the rest)")
    parser.add_argument("--i_tsv", dest="input_tsv", type=str, required=True, help="Path to the input tsv file")
    parser.add_argument("--i_dur", dest="input_dur", type=str, required=True, help="Path to the input duration tsv file")
    parser.add_argument("--o_train", dest="output_train", type=str, required=False, help="Path to the output train tsv file", default="train.tsv")
    parser.add_argument("--o_test", dest="output_test", type=str, required=False, help="Path to the output test tsv file", default="test.tsv")
    parser.add_argument("--o_val", dest="output_val", type=str, required=False, help="Path to the output val tsv file", default="val.tsv")
    parser.add_argument("--split_ratio", dest="split_ratio", type=float, nargs=2, required=False, help="Ratio to split the dataframe", default=[0.7, 0.1])
    args = parser.parse_args()
    main(args)