"""Call train_sed.py with the default config file and GPU 1. and then call the clean_process.py script to kill the process."""

import argparse
import os
import nni
import yaml

def main(command="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1"):
    try:
        
        print(f'Loading the config file from {command}')
        # Parse the command and load the config file
        command = command.split(" ")
        config_file = command[command.index("--conf_file")+1]
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Get the parameters from nni
        params = nni.get_next_parameter()
        print(f"Parameters from nni: {params}")
        
        # Update the config file with the new parameters
        for key, value in params.items():
            for config_key in config["net"].keys():
                if key == config_key:
                    config["net"][key] = value
            for config_key in config["feats"].keys():
                if key == config_key:
                    config["feats"][key] = value
                    
        # Save the new config file as 'config_nni.yaml'
        with open("config_nni.yaml", "w") as f:
            yaml.dump(config, f)
            
        print("New config file saved as 'config_nni.yaml'")
            
        # Update the command
        command[command.index("--conf_file")+1] = "config_nni.yaml"
        command = " ".join(command)                    
        os.system(command) # training run
    
    except Exception as e:
        print(f"Error: {e}")    
    
    os.system(f"python3 utils/clean_process.py --target_command '{command}'") # kill the process
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1")
    args = parser.parse_args()
    main(args.command)