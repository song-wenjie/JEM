import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torch.autograd import Variable
import cv2

n_iter = int(sys.argv[1])  # the maximum iteration step of SGLD
stepsize = float(sys.argv[2]) # the stepsize of SGLD
dataset_type = sys.argv[3] # hflung, covid, both the two sets
device = sys.argv[4]

batchsize = 160
unlabeled_batchsize = 80
labeled_batchsize = 80

AKGC417L = np.load('spectrum_correction/official/AKGC417L.npy')
Meditron = np.load('spectrum_correction/official/Meditron.npy')
Litt3200 = np.load('spectrum_correction/official/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/official/LittC2SE.npy')
HF_Type = np.load('spectrum_correction/official/HF_Type.npy')
Littmann = np.load('spectrum_correction/official/Littmann.npy')
crowdsource = np.load('spectrum_correction/official/crowdsource.npy')


def spectrum(audio,sr, device_index):
    D = np.abs(librosa.stft(audio, n_fft=256, hop_length=64)) ** 2
    while D.shape[1] < 320:
        D = np.concatenate((D, D), axis=1)
    D = D[:, 0:320]
    D = D.T
    if device_index == 'L':
        D = D * AKGC417L
    elif device_index == 'n':
        D = D * Meditron
    elif device_index == '0':
        D = D * Litt3200
    elif device_index == 'E':
        D = D * LittC2SE
    elif device_index =='s':
        D = D * Littmann
    elif device_index == 't':
        D = D * HF_Type
    elif device_index == 'c':
        D = D * crowdsource
    D = D.T
    mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel)
    mel = cv2.cvtColor(mel, cv2.COLOR_GRAY2BGR)
    mel = mel.transpose(2, 0, 1)
    return mel


if dataset_type=='both':
    filenames = glob.glob('dataset/covid/*')
    x_unlabeled_a = np.empty(shape=(len(filenames), 3, 64, 320))
    for i in range(len(filenames)):
        wav, sr = librosa.load(filenames[i], sr=None)
        x_unlabeled_a[i, :, :, :] = spectrum(wav, sr, 'c')
    x_unlabeled_a = x_unlabeled_a - np.mean(x_unlabeled_a, axis=0)
    x_unlabeled_a = x_unlabeled_a / np.std(x_unlabeled_a, axis=0)

    filenames = glob.glob('dataset/HF_Lung_V1/*')
    x_unlabeled_b = np.empty(shape=(len(filenames), 3, 64, 320))
    for i in range(len(filenames)):
        wav, sr = librosa.load(filenames[i], sr=None)
        name = str(filenames[i]).split('/')
        name = name[-1]
        x_unlabeled_b[i, :, :, :] = spectrum(wav, sr, name[0])
    x_unlabeled_b = x_unlabeled_b - np.mean(x_unlabeled_b, axis=0)
    x_unlabeled_b = x_unlabeled_b / np.std(x_unlabeled_b, axis=0)

    x_unlabeled = np.concatenate((x_unlabeled_a, x_unlabeled_b), axis=0)
    x_unlabeled_a = []
    x_unlabeled_b = []
elif dataset_type=='hflung':
    filenames = glob.glob('dataset/HF_Lung_V1/*')
    x_unlabeled = np.empty(shape=(len(filenames), 3, 64, 320))
    for i in range(len(filenames)):
        wav, sr = librosa.load(filenames[i], sr=None)
        name = str(filenames[i]).split('/')
        name = name[-1]
        x_unlabeled[i, :, :, :] = spectrum(wav, sr, name[0])
    x_unlabeled = x_unlabeled - np.mean(x_unlabeled, axis=0)
    x_unlabeled = x_unlabeled / np.std(x_unlabeled, axis=0)
else:
    filenames = glob.glob('dataset/covid/*')
    x_unlabeled = np.empty(shape=(len(filenames), 3, 64, 320))
    for i in range(len(filenames)):
        wav, sr = librosa.load(filenames[i], sr=None)
        x_unlabeled[i, :, :, :] = spectrum(wav, sr, 'c')
    x_unlabeled = x_unlabeled - np.mean(x_unlabeled, axis=0)
    x_unlabeled = x_unlabeled / np.std(x_unlabeled, axis=0)


# official splitting--test set
filenames = np.concatenate((glob.glob('dataset/official/test/crackle/*'),
                            glob.glob('dataset/official/test/wheeze/*'),
                            glob.glob('dataset/official/test/both/*'),
                            glob.glob('dataset/official/test/normal/*')))
index = np.random.permutation(np.arange(len(filenames)))
filenames = np.array(filenames)[index]
x_test = np.empty(shape=(len(filenames),3,64,320))
y_test = []
for i in range(len(filenames)):
    name = str(filenames[i]).split('/')
    if name[-2]=='crackle':
        y_test.append(0)
    elif name[-2]=='wheeze':
        y_test.append(1)
    elif name[-2]=='both':
        y_test.append(2)
    else:
        y_test.append(3)
    name = name[-1]
    wav, sr = librosa.load(filenames[i], sr=None)
    x_test[i,:,:,:] = spectrum(wav, sr, name[-8])


filenames = np.concatenate((glob.glob('dataset/official/train/crackle/*'),
                            glob.glob('dataset/official/train/wheeze/*'),
                            glob.glob('dataset/official/train/both/*'),
                            glob.glob('dataset/official/train/normal/*')))
x_train = np.empty(shape=(len(filenames),3,64,320))
y_train = []
for i in range(len(filenames)):
    name = str(filenames[i]).split('/')
    if name[-2] == 'crackle':
        y_train.append(0)
    elif name[-2] == 'wheeze':
        y_train.append(1)
    elif name[-2] == 'both':
        y_train.append(2)
    else:
        y_train.append(3)
    name = name[-1]
    wav,sr = librosa.load(filenames[i],sr=None)
    x_train[i,:,:,:] = spectrum(wav,sr, name[-8])


m = np.mean(x_train, axis=0)
x_train = x_train-m
std = np.std(x_train, axis=0)
x_train = x_train/std
x_test = x_test-m
x_test = x_test/std


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mobileVit = torch.load('model_structure.pt')
        self.mobileVit.load_state_dict(torch.load('../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt', map_location=device))
        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.W = torch.nn.Parameter(torch.randn(960, 4), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, img):
        x = self.mobileVit(img)
        x = x.squeeze()
        x = torch.mm(x, self.W)
        return x


class MainModel(nn.Module):
    def __init__(self, encodernet):
        super(MainModel, self).__init__()
        self.encodernet = encodernet

    def forward(self, img, label, sgld_img):
        prediction = self.encodernet(img)
        unlabeled_prediction, labeled_prediction = torch.split(prediction, unlabeled_batchsize, dim=0)
        loss1 = F.cross_entropy(input=labeled_prediction, target=label)

        logsumexp = torch.logsumexp(prediction, dim=1)

        sgld_prediction = self.encodernet(sgld_img)
        sgld_logsumexp = torch.logsumexp(sgld_prediction, dim=1)

        loss2 = torch.mean(sgld_logsumexp - logsumexp)
        return loss1, loss2


encoder = Encoder().to(device)
mainmodel = MainModel(encoder).to(device)
optimizer = optim.Adam(mainmodel.parameters(), lr=1e-3)
history_score = 0


for epoch in range(50):
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    index = np.random.permutation(np.arange(len(x_train)))
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]
    max_batch = int(len(x_train) / labeled_batchsize)
    encoder.train()
    mainmodel.train()
    for batch in range(max_batch):
        batch_labeled_img = x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_label = y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_label = torch.tensor(batch_label).to(device)
        batch_unlabeled_img = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_img = np.concatenate((batch_unlabeled_img, batch_labeled_img), axis=0)
        batch_img = torch.from_numpy(batch_img).float().to(device)
        particles = torch.clone(batch_img)
        alpha = 0.99
        fudge_factor = 1e-12
        historical_grad = 0
        for iter in range(n_iter):
            particles = Variable(particles, requires_grad=True)
            prediction = encoder(particles)
            lnp = torch.sum(torch.log(torch.sum(torch.exp(prediction), dim=1)), dim=0)
            lnp.backward()
            if iter == 0:
                historical_grad = historical_grad + particles.grad ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (particles.grad ** 2)
            adj_grad = torch.divide(particles.grad,  torch.sqrt(historical_grad+fudge_factor))
            particles = particles + (stepsize/2) * adj_grad + stepsize * torch.rand_like(particles)
            particles = particles.detach()

        optimizer.zero_grad()
        batch_loss1, batch_loss2 = mainmodel(batch_img, batch_label, particles)
        loss = batch_loss1 + batch_loss2
        loss.backward()
        optimizer.step()

    encoder.eval()
    mainmodel.eval()
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    max_batch = int(len(x_test) / batchsize)
    for batch in range(max_batch):
        batch_img = x_test[batch * batchsize: batch * batchsize + batchsize]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_test[batch * batchsize:batch * batchsize + batchsize]
        with torch.no_grad():
            batch_pred = encoder(batch_img)
        batch_pred = torch.argmax(batch_pred, dim=1)
        for i in range(batchsize):
            result[batch_label[i]][batch_pred[i]] += 1
    batch_img = x_test[max_batch * batchsize:]
    batch_img = torch.from_numpy(batch_img).float().to(device)
    batch_label = y_test[max_batch * batchsize:]
    with torch.no_grad():
        batch_pred = encoder(batch_img)
    batch_pred = torch.argmax(batch_pred, dim=1)
    for i in range(len(x_test) - max_batch * batchsize):
        result[batch_label[i]][batch_pred[i]] += 1

    acc = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(x_test)
    se = (result[0][0] + result[1][1] + result[2][2]) / (
            len(x_test) - result[3][0] - result[3][1] - result[3][2] - result[3][3])
    sp = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
    score = (se + sp) / 2

    if score > history_score:
        history_score = score
        print('epoch ' + str(epoch) + ':accuracy:' + str(acc) + ';sensitivity:' + str(se) + ';specificity:' + str(sp) + ';score:' + str(score))