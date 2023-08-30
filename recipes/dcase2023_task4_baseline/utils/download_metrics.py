# Download the metrics from the remote server with scp

import paramiko
import os
import argparse
import platform

def download_metrics(user, password, hostname, version, local_path, remote_path):
    """
    Download the metrics from the remote server with scp
    """
    print("--------------------------------------------------\n")
    # Establish an SSH connection
    print("Establishing an SSH connection to " + hostname + "\n")
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=user, password=password)
    print("SSH connection established\n")
    
    
    # Go to the remote path
    print("Command: cd " + remote_path + "\n")
    ssh_client.exec_command('cd ' + remote_path)

    # If no version is specified, determine the last version on the remote server
    # foldername = 'version_' + version
    sftp_client = ssh_client.open_sftp()
    if version == 'last_version':
        folder_list = sftp_client.listdir(remote_path)
        
        # Filter folder names starting with "version_"
        version_folders = [folder for folder in folder_list if folder.startswith("version_")]
        
        # Sort the folder names in descending order
        sorted_folders = sorted(version_folders, key=lambda x: int(x.split('_')[-1]), reverse=True)
        
        # Get the last version folder
        last_version = sorted_folders[0] if sorted_folders else None
        # Convert to string
        version = str(last_version) + '/'
        print('Last version: ' + version + '\n')

    # Copy the metrics to the local folder from the local machine
    # Test if the local machine is Windows or Linux
    if platform.system() == 'Windows':
        # Windows
        print("Command: scp -r " + user + "@" + hostname + ":" + remote_path + version + " " + local_path + "\n")
        os.system("scp -r " + user + "@" + hostname + ":" + remote_path + version + " " + local_path)
    else:
        # Linux
        print("Command: sshpass -p " + password + " scp -r " + user + "@" + hostname + ":" + remote_path + version + " " + local_path + "\n")
        os.system("sshpass -p " + password + " scp -r " + user + "@" + hostname + ":" + remote_path + version + " " + local_path)
    print("Metrics downloaded to " + local_path + "\n")
    
    # Close the SSH connection
    ssh_client.close()
    
    print('Done, connection closed\n')
    print("--------------------------------------------------\n")
    
parser = argparse.ArgumentParser(description='download the metrics of a specified version from the remote server')

# The user name with which to connect to the remote server
parser.add_argument('--user', type=str, default='user', help='user name')

# The password of the remote server for the user
parser.add_argument('--password', type=str, default='password', help='password')

# The remote address and path
remote_hostname = 'lenovo-GPU.ave.kth.se'
remote_path = '/srv/shared_data/DESED_task/recipes/dcase2023_task4_baseline/exp/2023_baseline/'
parser.add_argument('--hostname', type=str, default=remote_hostname, help='hostname of the remote server')
parser.add_argument('--remote_path', type=str, default=remote_path, help='path on the remote server')

# The local path to which to download the metrics
home = os.path.join(os.path.expanduser('~'), 'Downloads')
parser.add_argument('--path', type=str, default=home, help='path to the local folder')

# The version of the metrics to download
parser.add_argument('--version', type=str, default='last_version', help='version of the metrics to download')

args = parser.parse_args()


download_metrics(args.user, args.password, args.hostname, args.version, args.path, args.remote_path)