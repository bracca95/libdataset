import os
import sys
import torch
import random
import numpy as np

from src.models.model_utils import Model
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.train_test.routine import TrainTestExample
from src.train_test.proto_routine import ProtoRoutine
from src.datasets.defectviews import DefectViews
from src.datasets.dataset_utils import DatasetBuilder
from src.utils.config_parser import Config
from src.utils.tools import Logger
from config.consts import General as _CG

SEED = 1234         # with the first protonet implementation I used 7

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


if __name__=="__main__":
    try:
        config = Config.deserialize("config/config.json")
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    try:
        dataset = DatasetBuilder.load_dataset(config)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # compute mean and variance of the dataset if not done yet
    if config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        DefectViews.compute_mean_std(dataset, config)
        sys.exit(0)

    # store config so that you know what you have run :)
    config.serialize(os.path.join(os.getcwd(), "output"), "out_config.json")

    ## TODO: Create model instantiator
    ## TODO: online augmentation
    ## TODO: tensorboard
    # train, (val), test split
    model = ProtoNet().to(_CG.DEVICE)
    model_path = os.path.join(os.getcwd(), "output/best_model.pth")
    
    # split dataset
    subsets_dict = DefectViews.split_dataset(dataset, [0.8])
    
    # train/test
    routine = ProtoRoutine(model, dataset, subsets_dict)
    if not os.path.exists(model_path) or (os.path.exists(model_path) and not os.path.isfile(model_path)):
        routine.train(config)
    else:
        Logger.instance().warning("A model exists in output dir. Remove it or rename it if you want to train again.")
    routine.test(config, model_path)
