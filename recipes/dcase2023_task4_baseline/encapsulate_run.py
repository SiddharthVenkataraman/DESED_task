"""Call train_sed.py with the default config file and GPU 1. and then call the clean_process.py script to kill the process."""

import argparse
import os

def main(command="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1"):
    os.system(command) # training run
    os.system(f"python3 utils/clean_process.py --target_command '{command}'") # kill the process
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1")
    args = parser.parse_args()
    main(args.command)