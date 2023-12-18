import psutil
import os
import signal
import argparse
import psutil
import os
import signal
import argparse


def kill_target_process(target_command):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            # Check if the process command matches the target command
            if ' '.join(proc.info['cmdline']) == target_command:
                print(f"Found target process with PID: {proc.info['pid']}")

                # Kill the process
                os.kill(proc.info['pid'], signal.SIGKILL)
                # verify that it is dead
                assert proc.status() == psutil.STATUS_ZOMBIE
                print(f"Killed process with PID: {proc.info['pid']}")

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Ignore processes that no longer exist or can't be accessed

def main(target_command="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1"):
    kill_target_process(target_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_command", type=str, default="python3 train_sed.py --conf_file confs/default_SV.yaml --gpus 1")
    args = parser.parse_args()
    main(args.target_command)