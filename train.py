import os
import math
import random
import time
import logging
import warnings
logging.basicConfig(level = logging.INFO, filename = 'dev.log', filemode = 'w', format = "%(asctime)s [%(filename)s:%(lineno)d - %(levelname)s ] %(message)s")
warnings.filterwarnings('ignore')

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import dataset
import model

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)

class Trainer(object):
    def __init__(self, path):
        f = open(path)
        opts = yaml.load(f, Loader = yaml.CLoader)
        f.close()

        self.opts = opts

        if os.path.exists(self.opts['resume']):
            self.log_time = self.opts['resume'].split('/')[1]
        else:
            self.log_time = time.asctime(time.localtime(time.time())).replace(' ', '_')

        #  self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')

        self.trainset = dataset.EmbeddingTrainDataset(opts['dataset'])
        self.enrollset = dataset.EmbeddingEnrollDataset(opts['dataset'])
        self.n_spks = opts['n_spks']
        self.n_utts = opts['n_utts']
        milestones = opts['milestones']
        batch_sampler = dataset.BalancedBatchSampler(self.trainset.labels, len(self.trainset), opts['n_spks'], opts['n_utts'])
        self.train_loader = DataLoader(self.trainset, batch_sampler = batch_sampler, num_workers = 16, pin_memory = True)
        self.eval_path = opts['dataset']['eval_path']

        self.backend = model.AttentionAggregation(self.opts['model']).to(self.device)

        self.optimizer = optim.SGD(self.backend.parameters(), opts['lr'], momentum = 0.9, weight_decay = 0.00001)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = milestones, gamma = 0.1)
        self.criterion = nn.BCELoss(reduction = 'sum')

        self.epochs = opts['epochs']
        self.current_epoch = 0
        self.grad_clip_threshold = 20

        if os.path.exists(self.opts['resume']):
            self.load(self.opts['resume'])
            print("Load model from {}".format(self.log_time))
        else:
            print("Start new training {}".format(self.log_time))


    def train(self):
        os.makedirs('exp/{}'.format(self.log_time), exist_ok = True)
        start_epoch = self.current_epoch
        for epoch in range(start_epoch + 1, self.epochs + 1):
            self.current_epoch = epoch
            self.train_epoch()
            self.lr_scheduler.step()

    def train_epoch(self):
        self.backend.train()
        sum_loss, sum_samples = 0, 0
        progress_bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, labels) in progress_bar:
            data = data.to(self.device) # spks * utts, 1, dimension
            data = data.view(self.n_spks, self.n_utts, -1)

            self.optimizer.zero_grad()
            scores, labels = self.backend(data)
            loss = self.criterion(scores, labels)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.backend.parameters(), self.grad_clip_threshold
            )
            logging.info("grad norm={}".format(grad_norm))

            if math.isnan(grad_norm):
                logging.warning("grad norm is nan. Do not update model.")
            else: 
                self.optimizer.step()

            sum_samples += self.n_spks * self.n_utts * self.n_spks
            sum_loss += loss.item()
            progress_bar.set_description(
                        'Train Epoch: {:3d} [{:4d}/{:4d} ({:3.3f}%)] Loss: {:.16f}'.format(
                            self.current_epoch, batch_idx + 1, len(self.train_loader),
                            100. * (batch_idx + 1) / len(self.train_loader), sum_loss / sum_samples
                            )
                        )
        self.save()

    def enroll(self):
        self.backend.eval()
        os.makedirs('./exp/{}/enroll'.format(self.log_time), exist_ok = True)
        with torch.no_grad():
            for embeddings, spk in tqdm(self.enrollset()):
                center = self.backend.center(embeddings.to(self.device))
                path = os.path.join('./exp/{}/enroll/'.format(self.log_time), spk) + '-enroll.npy'
                np.save(path, center.cpu().numpy())    

    def test(self):
        self.backend.eval()
        y_true = []
        y_pred4w = []
        y_pred = []
        count = 0
        enroll_dir = './exp/{}/'.format(self.log_time)
        wf = open('./exp/{}/backend_scores.txt'.format(self.log_time), 'w')
        with torch.no_grad():
            with open('../task.txt', 'r') as f:
                for line in tqdm(f):
                    count += 1
                    line = line.rstrip()
                    true_score, utt1, utt2 = line.split(' ')
                    y_true.append(eval(true_score))
                    s_utt1 = np.load(os.path.join(enroll_dir, utt1.replace('.wav', '.npy')))
                    s_utt2 = np.load(os.path.join(self.eval_path, utt2.replace('.wav', '.npy')))
                    score = self.backend.test(torch.from_numpy(s_utt1).to(self.device), torch.from_numpy(s_utt2).to(self.device))
                    line = utt1 + ' ' + utt2 + ' ' + str(score.item()) + '\n'
                    wf.write(line)
        wf.close()

    def save(self, filename = None):
        if filename is None:
            torch.save({'epoch': self.current_epoch, 'state_dict': self.backend.state_dict(), 'criterion': self.criterion,
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optimizer.state_dict()},
                        'exp/{}/net_{}.pth'.format(self.log_time, self.current_epoch))
        else:
            torch.save({'epoch': self.current_epoch, 'state_dict': self.backend.state_dict(), 'criterion': self.criterion,
                        'lr_scheduler': self.lr_scheduler.state_dict(), 'optimizer': self.optimizer.state_dict()},
                        'exp/{}/{}'.format(self.log_time, filename))

    def load(self, resume):
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(resume, map_location = map_location)
        self.backend.load_state_dict(ckpt['state_dict'])
        if 'criterion' in ckpt:
            self.criterion = ckpt['criterion']
        if 'lr_scheduler' in ckpt:
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.current_epoch = ckpt['epoch']

if __name__ == '__main__':
    trainer = Trainer('./config.yaml')
    trainer.train()
    trainer.enroll()
    trainer.test()
