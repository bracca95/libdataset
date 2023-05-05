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

    def train(self, config: Config):

        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        train_label_list = [self.dataset[idx][1] for idx in self.train_info.subset.indices]
        train_sampler = PrototypicalBatchSampler(
            train_label_list,
            config.fsl.train_n_way,
            config.fsl.train_k_shot_s + config.fsl.train_k_shot_q,
            config.fsl.episodes
        )
        trainloader = DataLoader(self.train_info.subset, batch_sampler=train_sampler)
        
        valloader = None
        if self.val_info.subset is not None:
            val_label_list = [self.dataset[idx][1] for idx in self.val_info.subset.indices]
            val_sampler = PrototypicalBatchSampler(
                val_label_list,
                config.fsl.test_n_way,
                config.fsl.test_k_shot_s + config.fsl.test_k_shot_q,
                config.fsl.episodes
            )
            valloader = DataLoader(self.val_info.subset, batch_sampler=val_sampler)
        
        optim = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim,
            gamma=self.lr_scheduler_gamma,
            step_size=self.lr_scheduler_step
        )

        best_state: Optional[dict] = None
        if valloader is None:
            best_state = None
        
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
            print('=== Epoch: {} ==='.format(epoch))
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
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
            
            # if validation is required
            if valloader is None:
                # goto next epoch
                continue
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
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(self.model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = self.model.state_dict()

        torch.save(self.model.state_dict(), last_model_path)

        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(out_folder, name + '.txt'), locals()[name])

        return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

    def test(self, config: Config):

        if config.fsl is None:
            raise ValueError(f"missing field `fsl` in config.json")

        test_label_list = [self.dataset[idx][1] for idx in self.test_info.subset.indices]
        test_sampler = PrototypicalBatchSampler(
            test_label_list,
            config.fsl.test_n_way,
            config.fsl.test_k_shot_s + config.fsl.test_k_shot_q,
            config.fsl.episodes
        )

        testloader = DataLoader(self.test_info.subset, batch_sampler=test_sampler)
        
        avg_acc = list()
        for epoch in range(10):
            test_iter = iter(testloader)
            for batch in test_iter:
                x, y = batch
                x, y = x.to(_CG.DEVICE), y.to(_CG.DEVICE)
                model_output = self.model(x)
                _, acc = loss_fn(model_output, target=y, n_support=config.fsl.test_k_shot_s)
                avg_acc.append(acc.item())
        avg_acc = np.mean(avg_acc)
        print('Test Acc: {}'.format(avg_acc))

        return avg_acc
#######



def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write(f"{item}\n")





def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    avg_acc = list()
    for epoch in tqdm(range(10)):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=args.num_support_val)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(args):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(args.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(args=options,
         test_dataloader=test_dataloader,
         model=model)


if __name__ == '__main__':

    # train and test datasets
    train_test_split = int(len(dataset)*0.8)
    trainset, testset = random_split(dataset, [train_test_split, len(dataset) - train_test_split])

    # extra info needed
    train_label_list = [dataset[idx][1] for idx in trainset.indices]
    test_label_list = [dataset[idx][1] for idx in testset.indices]

    Logger.instance().debug(f"samples per class: { {dataset.idx_to_label[i]: train_label_list.count(i) for i in set(train_label_list)} }")
    Logger.instance().debug(f"samples per class: { {dataset.idx_to_label[i]: test_label_list.count(i) for i in set(test_label_list)} }")
    
    train_sampler = init_sampler(args, train_label_list, 'train')
    test_sampler = init_sampler(args, test_label_list, 'test')
    
    trainloader = DataLoader(trainset, batch_sampler=train_sampler)
    testloader = DataLoader(testset, batch_sampler=test_sampler)

    model = init_protonet(args)
    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)

    pre_trained_model_path = os.path.join(args.experiment_root, "best_model.pth")
    if not os.path.exists(pre_trained_model_path):
        Logger.instance().debug("No model found, training mode ON!")
        res = train(args=args,
                    tr_dataloader=trainloader,
                    valloader=trainloader,
                    model=model,
                    optim=optim,
                    lr_scheduler=lr_scheduler)
        
        best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    else:
        Logger.instance().debug("Model found! Testing on your dataset!")
        model.load_state_dict(torch.load(pre_trained_model_path))
        test(args=args, test_dataloader=testloader, model=model)