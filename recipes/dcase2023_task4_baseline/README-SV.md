# Procedure to load and run DCASE2023 Baseline model for Task 4
The following steps assume the user is running a Linux system.
Please refer to [Link](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline) for more details.

## Obtain code

Clone Github repository `git@github.com:SiddharthVenkataraman/DESED_task.git`
- Add SSH key to deploy keys
- Can also fork this repository to personal GitHub account, and use that independently.

## Install libraries

Run `DESED_task/recipes/dcase2023_task4_baseline/conda_create_environment.sh` **line by line**
NOTE: Need to be in folder `DESED_task/recipes/dcase2023_task4_baseline` when running the above commands!
NOTE: If installing on CPU-only machine (i.e., cannot use CUDA-compatible pytorch), **DO NOT** run line #3 (conda install ... pytorch-cuda=11.7 ...).
Instead run the following command:
`conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Enable CUDA
If the computer has a CUDA-compatible GPU, ensure that the latest drivers are installed.
- A quick way to find the graphics card is to use `lspci | grep 'VGA'`
- Check if this card is CUDA-compatible by checking, e.g. [Wikipedia](https://en.wikipedia.org/wiki/CUDA)
- Check if the latest nvidia graphics are installed.
	- E.g. in Ubuntu, go to "Softward & Updates", "Additional Drivers", and choose the latest NVIDIA driver, such as "nvidia-driver-535"
- Check if the installation was successful by running the following commands:
```
conda activate dcase2023
python -c "import torch;print(torch.cuda.is_available())"
```
	- If output is True, then NVIDIA driver installation was successful.

### Additional installations
Do not run these commands unless running `python train_sed.py` throws errors.

```
sudo apt install ffmpeg

conda activate dcase2023

# Install psds-eval
pip install cython
pip install psds-eval

# Install sed-scores_eval
sudo apt install gcc
sudo apt install g++
pip install git+https://github.com/fgnt/sed_scores_eval.git

```
- Import "scores.py" from older version of sed_scores_eval and paste it in [miniconda3/anaconda3]/envs/dcase2023/lib/python3.8/site-packages/sed_scores_eval/utils/
Can get this "scores.py" from Siddharth

# Steps before running train_sed.py
1. Run `preprocess_tsv_files-SV.py` once, to ensure missing DCASE data will not be used during training

2. Run `train_sed.py` with the following parameters to check if everything works
`python train_sed.py --fast_dev_run --gpus 1`  if CUDA-compatible GPU is available and installed
or
`python train_sed.py --fast_dev_run --gpus 0`  otherwise

