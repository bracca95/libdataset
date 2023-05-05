import os
import sys
import torch
import numpy as np

from tqdm import tqdm
from torch import nn
from typing import Optional
from torch.utils.data import random_split, DataLoader

from src.datasets.defectviews import DefectViews
from src.models.FSL.ProtoNet.proto_batch_sampler import PrototypicalBatchSampler
from src.models.FSL.ProtoNet.proto_loss import prototypical_loss as loss_fn
from src.models.FSL.ProtoNet.protonet import ProtoNet
from src.utils.tools import Tools, Logger
from src.utils.config_parser import Config
from src.datasets.defectviews import DefectViews
from src.train_test.routine import TrainTest
from config.consts import General as _CG
from config.consts import SubsetsDict


class ProtoRoutine(TrainTest):

    def __init__(self, model: nn.Module, dataset: DefectViews, subsets_dict: SubsetsDict):
        super().__init__(model, dataset, subsets_dict)
        self.learning_rate = 0.001
        self.lr_scheduler_gamma = 0.5
        self.lr_scheduler_step = 20

    def init_loader(self, config: Config, split_set: str):
        current_subset = self.get_subset_info(split_set)
        if current_subset.subset is None:
            return None
        
        label_list = [self.dataset[idx][1] for idx in current_subset.subset.indices]
        sampler = PrototypicalBatchSampler(
            label_list,
            config.fsl.train_n_way if split_set == self.train_str else config.fsl.test_n_way,
            config.fsl.train_k_shot_s + config.fsl.train_k_shot_q if split_set == self.train_str else config.fsl.test_k_shot_s + config.fsl.test_k_shot_q,
            config.fsl.episodes
        )
        return DataLoader(current_subset.subset, batch_sampler=sampler)
    
    def save_list_to_file(self, path, thelist):
        with open(path, 'w') as f:
            for item in thelist:
                f.write(f"{item}\n")

    def train(self, config: Config):
        Logger.instance().debug("Start training")
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        trainloader = self.init_loader(config, self.train_str)
        valloader = self.init_loader(config, self.val_str)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.lr_scheduler_gamma,
            step_size=self.lr_scheduler_step
        )
        
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        best_acc = 0

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        best_model_path = os.path.join(out_folder, 'best_model.pth')
        last_model_path = os.path.join(out_folder, 'last_model.pth')

        for epoch in range(config.epochs):
            Logger.instance().debug(f"=== Epoch: {epoch} ===")
            tr_iter = iter(trainloader)
            self.model.train()
            for batch in tqdm(tr_iter):
                optim.zero_grad()
                x, y = batch
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                loss, acc = loss_fn(model_output, target=y, n_support=config.fsl.train_k_shot_s)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
            
            avg_loss = np.mean(train_loss[-config.fsl.episodes:])
            avg_acc = np.mean(train_acc[-config.fsl.episodes:])
            lr_scheduler.step()
            Logger.instance().debug(f"Avg Train Loss: {avg_loss}, Avg Train Acc: {avg_acc}")
            
            # if validation is required
            if valloader is not None:
                Logger.instance().debug("Validating!")
                val_iter = iter(valloader)
                self.model.eval()
                for batch in val_iter:
                    x, y = batch
                    x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                    model_output = self.model(x)
                    loss, acc = loss_fn(model_output, target=y, n_support=config.fsl.test_k_shot_s)
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())
                avg_loss = np.mean(val_loss[-config.fsl.episodes:])
                avg_acc = np.mean(val_acc[-config.fsl.episodes:])
                postfix = f" (Best)" if avg_acc >= best_acc else f" (Best: {best_acc})"
                Logger.instance().debug(f"Avg Val Loss: {avg_loss}, Avg Val Acc: {avg_acc}{postfix}")
            
            if avg_acc >= best_acc:
                Logger.instance().debug(f"Found the best model at epoch {epoch}!")
                torch.save(self.model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = self.model.state_dict()
                torch.save(self.model.state_dict(), best_model_path)

            torch.save(self.model.state_dict(), last_model_path)

        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            self.save_list_to_file(os.path.join(out_folder, name + '.txt'), locals()[name])

    def test(self, config: Config, model_path: str):
        Logger.instance().debug("Start testing")
        
        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")
        
        try:
            model_path = Tools.validate_path(model_path)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)

        self.model.load_state_dict(torch.load(model_path))
        testloader = self.init_loader(config, self.test_str)
        
        avg_acc = list()
        self.model.eval()
        for epoch in tqdm(range(10)):
            test_iter = iter(testloader)
            for batch in test_iter:
                x, y = batch
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                _, acc = loss_fn(model_output, target=y, n_support=config.fsl.test_k_shot_s)
                avg_acc.append(acc.item())
        
        avg_acc = np.mean(avg_acc)
        Logger.instance().debug(f"Test Acc: {avg_acc}")

        return avg_acc
