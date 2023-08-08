import argparse
import torch
import yaml
import os
import pandas as pd

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabeledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.sed_trainer import SEDTask4
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

from local.utils import (
    calculate_macs
)


import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from local.utils import generate_tsv_wav_durations
""" 
Script to load a particular model checkpoint, and use it for making predictions.
Run script ensuring:
    1. model checkpoint defined either with command line argument "--predict_from_checkpoint <path_to_cpkt_file>", or through yaml variable ['data']['predict_from_checkpoint']
    2. data to be predicted available in folder defined under ['data']['predict_folder_44k'] (or ['data']['predict_folder'] if data already in required sample rate) 
"""

def resample_data_generate_durations_for_predictions(config_data, prediction=False):
    """
    Script to resample data to appropriate sampling frequency given by config_data["fs"]
    Script copied from "train_sed.py" and reused to focus only on data meant for making predictions
    """
    if prediction:  # If only prediction, i.e., use checkpoint to predict and visualize results in config_data["pred_folder"]
        dsets = ["predict_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    # # Not sure if durations is required. Use below code if pred_dur folder is necessary
    # for base_set in ["predict"]:
    #     if not os.path.exists(config_data[base_set + "_dur"]) or computed:
    #         generate_tsv_wav_durations(
    #             config_data[base_set + "_folder"], config_data[base_set + "_dur"]
    #         )

def single_run(
    config,
    gpus,
    predict_model_state_dict=None,
    prediction=False,
    callbacks=None
):
    """
    Running sound event detection baseline
    Based on function in "train_sed.py"
    Args:
        config (dict): the dictionary of configuration params
        gpus (int): number of gpus to use
        predict_model_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to predict using the model.
    """

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    if prediction:
        prediction_data = UnlabeledSet(
            config["data"]["predict_folder"],
            encoder,
            pad_to=None,
            return_filename=True
        )
        prediction_dataset = torch.utils.data.ConcatDataset([prediction_data])

    ##### model definition  ############
    sed_student = CRNN(**config["net"])

    # calulate multiply–accumulate operation (MACs)
    # Required to ensure one can load the model state dict from pre-trained checkpoints.
    # Can remove this if the model was trained without such variables.
    macs, _ = calculate_macs(sed_student, config)
    print(f"---------------------------------------------------------------")
    print(f"Total number of multiply–accumulate operation (MACs): {macs}\n")

    if predict_model_state_dict is not None:

        train_dataset = None
        valid_dataset = None
        test_dataset = None  # Temporary fix?, since SEDTask4 does not have class for prediction data
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        fast_dev_run = False
        evaluation = False
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    # if fast_dev_run:
    flush_logs_every_n_steps = 1
    log_every_n_steps = 1
    limit_train_batches = 2
    limit_val_batches = 2
    limit_test_batches = 2
    n_epochs = 3
    # else:
    #     flush_logs_every_n_steps = 100
    #     log_every_n_steps = 40
    #     limit_train_batches = 1.0
    #     limit_val_batches = 1.0
    #     limit_test_batches = 1.0
    #     n_epochs = config["training"]["n_epochs"]

    if gpus == "0":
        accelerator = "cpu"
        devices = 1
    elif gpus == "1":
        accelerator = "gpu"
        devices = 1
    else:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )

    desed_training.load_state_dict(predict_model_state_dict)
    # TODO: Add console parameters "--only_predict", and "path_to_predictions" if predictions are required.
    trainer.predict(desed_training, prediction_dataset)
    # desed_training.predict_SV(['../../data/dcase/dataset/audio/validation/validation_16k/Y8ws1ligErwo_0.000_10.000.wav'])
    # trainer.test(desed_training)

def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/default_SV.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--predict_from_checkpoint",
        default=None,
        help="Predict and visualise results using the model specified"
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    if args.predict_from_checkpoint is not None:
        predict_from_checkpoint = args.predict_from_checkpoint
    else:
        predict_from_checkpoint = configs['data']['predict_from_checkpoint']

    prediction = True  # If --predict_from_checkpoint <path_to_.ckpt> gives as argument

    # If prediction required, load the ckeckpoint model passed as argument --predict_from_checkpoint <path_to_.ckpt>
    predict_model_state_dict = None
    if predict_from_checkpoint is not None:
        if torch.cuda.is_available():
            checkpoint = torch.load(predict_from_checkpoint)
        else:
            checkpoint = torch.load(predict_from_checkpoint, map_location=torch.device('cpu'))

        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {predict_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        predict_model_state_dict = checkpoint["state_dict"]

    resample_data_generate_durations_for_predictions(configs["data"], prediction)  # Requires predictions to be in 16k
    return configs, args, predict_model_state_dict, prediction


if __name__ == "__main__":
    # prepare run
    configs, args, predict_model_state_dict, prediction = prepare_run()

    # launch run
    single_run(
        configs,
        args.gpus,
        predict_model_state_dict,
        prediction
    )