"""
Executable python file that calls train_sed with the parameters obtained from nni
"""

import argparse
import os
import nni
import logging
import yaml


def main(args):
    """Main function"""
    
    # Print the current location
    print(os.getcwd())

    # Open the config file and create a copy of it
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_copy = config.copy()
    
    # Get the parameters from nni
    params = nni.get_next_parameter()
    logging.debug(params)
    
    # Update the config file with the new parameters
    for key, value in params.items():
        if key != "attention" and key != "activation":
            config_copy[key] = value
        elif key == "attention":
            if value == 1:
                config_copy["attention"] = True
            else:
                config_copy["attention"] = False
                
        elif key == "activation":
            if value == 1:
                config_copy["activation"] = "glu"
            else:
                config_copy["activation"] = "Relu"
    
    # Save the config file
    config_file = os.path.join(os.path.dirname(args.config), 'config_nni.yml')
    with open(config_file, 'w') as f:
        yaml.dump(config_copy, f)
        
    # Call train_sed with the new config file
    command = 'python3 train_sed.py --conf_file {}'.format(config_file) + ' --gpus 1' + ' --fast_dev_run'
    os.system(command)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, default="confs/config_SV.yaml",
                        help='Path to the config file to modify')
    args = parser.parse_args()
    main(args)