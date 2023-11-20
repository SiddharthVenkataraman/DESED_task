"""Takes a pandas datadrame and split it according to the split ratio [train, test]"""

from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

def split_df(df: pd.DataFrame, dur_df:pd.DataFrame, split_ratio: list, random_state: int = 42):
    """Splits a dataframe into train, test and val according to the split ratio [train, test]
    Args:
        df (pd.DataFrame): Dataframe to split
        dur_df (pd.DataFrame): Dataframe with the duration of each file
        split_ratio (list): Ratio to split the dataframe
        random_state (int, optional): Random state for the split. Defaults to 42.
    Returns:
        list: List of dataframes [train, test]
    """
    csv = df.copy()
    csv.filename = csv.filename.apply(lambda x: x.split("/")[-1])
    print(f"Total: {len(csv)} files")
    train, test = train_test_split(csv, test_size=split_ratio[1], random_state=random_state)
    print(f"Train: {len(train)} files, Test: {len(test)} files")
    train_dur = dur_df[dur_df.filename.isin(train.filename)]
    test_dur = dur_df[dur_df.filename.isin(test.filename)]
    print(f"Train duration: {train_dur.duration.sum()} s, Test duration: {test_dur.duration.sum()} s")
    return [(train, train_dur), (test, test_dur)]

 
def main(args):
    df = pd.read_csv(args.input_tsv, sep="\t")
    dur_df = pd.read_csv(args.input_dur, sep="\t")
    
    # Split dataframe
    split_dfs = split_df(df, dur_df, args.split_ratio)
    outputs = [args.output_train, args.output_test, args.output_val]
    for i, split_df in enumerate(split_dfs):
        split_df[0].to_csv(outputs[i], sep="\t", index=False)
        split_df[1].to_csv(outputs[i].replace(".tsv", "_duration.tsv"), sep="\t", index=False)
        print(f"Saved {outputs[i]}")
    
    # Create empty dataframe for placeholder
    placeholder = pd.DataFrame(columns=df.columns)
    placeholder.to_csv("placeholder.tsv", sep="\t", index=False)
        
if __name__ == '__main__':
    print("Splitting dataframe")
    parser = argparse.ArgumentParser(description="Split a dataframe into train, test and val according to the split ratio [train, test] (val is the rest)")
    parser.add_argument("--i_tsv", dest="input_tsv", type=str, required=True, help="Path to the input tsv file")
    parser.add_argument("--i_dur", dest="input_dur", type=str, required=True, help="Path to the input duration tsv file")
    parser.add_argument("--o_train", dest="output_train", type=str, required=False, help="Path to the output train tsv file", default="train.tsv")
    parser.add_argument("--o_test", dest="output_test", type=str, required=False, help="Path to the output test tsv file", default="test.tsv")
    parser.add_argument("--split_ratio", dest="split_ratio", type=float, nargs=2, required=False, help="Ratio to split the dataframe", default=[0.8, 0.2])
    args = parser.parse_args()
    main(args)