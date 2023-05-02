import sys
import torch
import random
import numpy as np

from torch.utils.data import random_split
from src.utils.config_parser import Config
from src.datasets import DefectViews, MNIST, BubblePoint, TTSet
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
        Logger.instance().error(e.args)
        sys.exit(-1)

    if config.dataset_type == "all":
        dataset = DefectViews(config.dataset_path, config.augment_offline, config.augment_online, config.crop_size)
    elif config.dataset_type == "binary":
        dataset = BubblePoint(config.dataset_path, config.augment_online, config.crop_size)
    else:
        Logger.instance.error("either `all` or `binary` for dataset_type")
        sys.exit(-1)
    
    # compute mean and variance of the dataset if not done yet
    if config.dataset_mean is None and config.dataset_std is None:
        Logger.instance().warning("No mean and std set: computing and storing values.")
        DefectViews.compute_mean_std(dataset, config)
        sys.exit(0)

    # if type(dataset) is MNIST:
    #     trainset, testset = dataset.get_train_test()
    #     trainset = TTSet(trainset)
    #     testset = TTSet(testset)
    # else:
    #     train_test_split = int(len(dataset)*0.8)
    #     train_subset, test_subset = random_split(dataset, [train_test_split, len(dataset) - train_test_split])
    #     trainset = TTSet(train_subset.dataset, train_subset.indices, dataset)
    #     testset = TTSet(test_subset.dataset, test_subset.indices, dataset)

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