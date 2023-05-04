import sys
import torch
import random
import numpy as np

from src.models.model_utils import Model
from src.train_test.routine import TrainTestExample
from src.datasets.defectviews import DefectViews
from src.datasets.dataset_utils import DatasetBuilder
from src.utils.config_parser import Config
from src.utils.tools import Logger

SEED = 1234

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

    # train, (val), test split
    model = Model()
    subsets_dict = DefectViews.split_dataset(dataset, [0.8])
    example = TrainTestExample(model, dataset, subsets_dict)

    example.train()
    example.test()

    # if config.mode == "mlp":
    #     Logger.instance().debug("running MLP")
    #     model = MLP(dataset.in_dim * dataset.in_dim, dataset.out_dim)
    # elif config.mode == "rescnn":
    #     Logger.instance().debug("running ResCNN")
    #     model = ResCNN(dataset.in_dim, dataset.out_dim)
    # elif config.mode == "cnn":
    #     Logger.instance().debug("running CNN")
    #     model = CNN(dataset.out_dim)
    # else:
    #     raise ValueError("either 'mlp' or 'cnn' or 'rescnn'")

    # if config.train:
    #     Logger.instance().debug("Starting training...")
    #     trainer = Trainer(trainset.tt_set, model)
    #     trainer.train(config)
    # else:
    #     Logger.instance().debug("Starting testing...")
    #     tester = Tester(testset, model, "checkpoints/model.pt")
    #     tester.test(config)

    # Logger.instance().debug("program terminated")