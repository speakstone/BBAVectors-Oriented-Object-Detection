import torch
import torch.nn as nn
import os
import numpy as np
import loss
from torch.utils.tensorboard import SummaryWriter
import func_utils

def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def __init__(self, dataset, num_classes, model, decoder, down_ratio):

        self.dataset = dataset
        self.dataset_phase = {'dota': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio


    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):
        datapth = args.checkpoint
        if not os.path.exists(datapth):
            os.makedirs(datapth)
        # 定义tensorboard存储
        self.writer = SummaryWriter(os.path.join(datapth, 'log'))

        # 定义其他训练参数
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = os.path.join(datapth, 'weights_'+args.dataset)
        start_epoch = 1
        
        # add resume part for continuing training when break previously, 10-16-2020
        if args.resume_train:
            self.model, self.optimizer, start_epoch = self.load_model(self.model, 
                                                                        self.optimizer, 
                                                                        args.resume_train, 
                                                                        strict=True)
        # end 

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}
        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []
        loss_index = 0
        mAP_max = 0
        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss, loss_list = self.run_epoch_accumulation(phase='train',
                                                   data_loader=dsets_loader['train'],
                                                   criterion=criterion,
                                                   acstps=args.accumulation_steps)
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch)

            # 将loss绘制tensorboard
            for index, loss_i in enumerate(loss_list):
                loss_index += 1
                self.writer.add_scalar('Loss', loss_i, loss_index)

            np.savetxt(os.path.join(save_path, 'train_loss.txt'), train_loss, fmt='%.6f')

            if epoch % 10 == 0:
                # 修改为每10个保存一次
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

            if 'test' in self.dataset_phase[args.dataset] and epoch%1==0:
                mAP = self.dec_eval(args, dsets['test'])
                # 将ap绘制tensorboard
                self.writer.add_scalar('mAP', mAP, epoch)
                ap_list.append(mAP)
                np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

                # 添加最优模型存储
                if mAP > mAP_max:
                    mAP_max = mAP
                    self.save_model(os.path.join(save_path, 'model_best.pth'),
                                    epoch,
                                    self.model,
                                    self.optimizer)

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        loss_list = []
        for data_dict in data_loader:
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
            # 存储loss输出
            loss_list.append(loss)
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        if phase == 'train':
            return epoch_loss, loss_list
        return epoch_loss

    def run_epoch_accumulation(self, phase, data_loader, criterion, acstps=1):
        if phase == 'train':
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        running_loss = 0.
        loss_list = []

        for index, data_dict in enumerate(data_loader):
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                with torch.enable_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict) / acstps
                    loss.backward()
                    if (index + 1) % acstps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    loss = criterion(pr_decs, data_dict)
            # 存储loss输出
            loss_list.append(loss * acstps)
            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader) * acstps
        print('{} loss: {}'.format(phase, epoch_loss))
        if phase == 'train':
            return epoch_loss, loss_list
        return epoch_loss

    def dec_eval(self, args, dsets):
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model,dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)
        return ap