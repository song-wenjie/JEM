import random
import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import sys

device = sys.argv[1]
dataset_type = sys.argv[2]
vad_option = sys.argv[3]
batchsize = 260
unlabeled_batchsize = 130
labeled_batchsize = 130
print('FixMatch')

AKGC417L = np.load('spectrum_correction/vad'+vad_option+'/AKGC417L.npy')
Meditron = np.load('spectrum_correction/vad'+vad_option+'/Meditron.npy')
Litt3200 = np.load('spectrum_correction/vad'+vad_option+'/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/vad'+vad_option+'/LittC2SE.npy')
HF_Type = np.load('spectrum_correction/vad'+vad_option+'/HF_Type.npy')
Littmann = np.load('spectrum_correction/vad'+vad_option+'/Littmann.npy')
crowdsource = np.load('spectrum_correction/vad'+vad_option+'/crowdsource.npy')


def augment(img, weak_frequency_masking_para=10, strong_frequency_masking_para=20, weak_time_masking_para=20, strong_time_masking_para=50):
    img1 = np.empty(shape=(unlabeled_batchsize, 3, 64, 320))
    img2 = np.empty(shape=(unlabeled_batchsize, 3, 64, 320))
    for i in range(unlabeled_batchsize):
        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=weak_frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=weak_time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        temp = np.reshape(temp, newshape=(3, 64, 320))
        img1[i, :, :, :] = temp

        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=strong_frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=strong_time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        temp = np.reshape(temp, newshape=(3, 64, 320))
        img2[i, :, :, :] = temp
    return img1,img2


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


filenames1 = np.concatenate((glob.glob('dataset/ICBHI/vad1/crackle/*'),
                            glob.glob('dataset/ICBHI/vad1/wheeze/*'),
                            glob.glob('dataset/ICBHI/vad1/both/*'),
                            glob.glob('dataset/ICBHI/vad1/normal/*')))
filenames2 = np.concatenate((glob.glob('dataset/ICBHI/vad2/crackle/*'),
                            glob.glob('dataset/ICBHI/vad2/wheeze/*'),
                            glob.glob('dataset/ICBHI/vad2/both/*'),
                            glob.glob('dataset/ICBHI/vad2/normal/*')))
filenames3 = np.concatenate((glob.glob('dataset/ICBHI/vad3/crackle/*'),
                            glob.glob('dataset/ICBHI/vad3/wheeze/*'),
                            glob.glob('dataset/ICBHI/vad3/both/*'),
                            glob.glob('dataset/ICBHI/vad3/normal/*')))
filenames4 = np.concatenate((glob.glob('dataset/ICBHI/vad4/crackle/*'),
                            glob.glob('dataset/ICBHI/vad4/wheeze/*'),
                            glob.glob('dataset/ICBHI/vad4/both/*'),
                            glob.glob('dataset/ICBHI/vad4/normal/*')))
filenames5 = np.concatenate((glob.glob('dataset/ICBHI/vad5/crackle/*'),
                            glob.glob('dataset/ICBHI/vad5/wheeze/*'),
                            glob.glob('dataset/ICBHI/vad5/both/*'),
                            glob.glob('dataset/ICBHI/vad5/normal/*')))

if vad_option=='1':
    test_filenames = filenames1
    vad_filenames = filenames2
    train_filenames = np.concatenate((filenames3, filenames4, filenames5))
elif vad_option=='2':
    test_filenames = filenames2
    vad_filenames = filenames3
    train_filenames = np.concatenate((filenames1, filenames4, filenames5))
elif vad_option=='3':
    test_filenames = filenames3
    vad_filenames = filenames4
    train_filenames = np.concatenate((filenames1, filenames2, filenames5))
elif vad_option=='4':
    test_filenames = filenames4
    vad_filenames = filenames5
    train_filenames = np.concatenate((filenames1, filenames2, filenames3))
elif vad_option=='5':
    test_filenames = filenames5
    vad_filenames = filenames1
    train_filenames = np.concatenate((filenames2, filenames3, filenames4))


index = np.random.permutation(np.arange(len(test_filenames)))
test_filenames = np.array(test_filenames)[index]
x_test = np.empty(shape=(len(test_filenames),3,64,320))
y_test = []
for i in range(len(test_filenames)):
    name = str(test_filenames[i]).split('/')
    if name[-2]=='crackle':
        y_test.append(0)
    elif name[-2]=='wheeze':
        y_test.append(1)
    elif name[-2]=='both':
        y_test.append(2)
    else:
        y_test.append(3)
    name = name[-1]
    wav, sr = librosa.load(test_filenames[i], sr=None)
    x_test[i,:,:,:] = spectrum(wav, sr, name[-8])


index = np.random.permutation(np.arange(len(vad_filenames)))
vad_filenames = np.array(vad_filenames)[index]
x_vad = np.empty(shape=(len(vad_filenames),3,64,320))
y_vad = []
for i in range(len(vad_filenames)):
    name = str(vad_filenames[i]).split('/')
    if name[-2]=='crackle':
        y_vad.append(0)
    elif name[-2]=='wheeze':
        y_vad.append(1)
    elif name[-2]=='both':
        y_vad.append(2)
    else:
        y_vad.append(3)
    name = name[-1]
    wav, sr = librosa.load(vad_filenames[i], sr=None)
    x_vad[i,:,:,:] = spectrum(wav, sr, name[-8])

# train set
x_train = np.empty(shape=(len(train_filenames),3,64,320))
y_train = []
for i in range(len(train_filenames)):
    name = str(train_filenames[i]).split('/')
    if name[-2] == 'crackle':
        y_train.append(0)
    elif name[-2] == 'wheeze':
        y_train.append(1)
    elif name[-2] == 'both':
        y_train.append(2)
    else:
        y_train.append(3)
    name = name[-1]
    wav,sr = librosa.load(train_filenames[i],sr=None)
    x_train[i,:,:,:] = spectrum(wav,sr, name[-8])


m = np.mean(x_train, axis=0)
x_train = x_train-m
std = np.std(x_train, axis=0)
x_train = x_train/std
x_test = x_test-m
x_test = x_test/std
x_vad = x_vad-m
x_vad = x_vad/std


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

    def forward(self, labeled_img, weak_img, strong_img, label):
        labeled_prediction = self.encodernet(labeled_img)
        labeled_loss = F.cross_entropy(labeled_prediction, label)

        weak_prediction = self.encodernet(weak_img)
        weak_prediction = F.softmax(weak_prediction, dim=1)
        weak_prediction = torch.where(weak_prediction > 0.95, torch.ones_like(weak_prediction), torch.zeros_like(weak_prediction))
        sum_weak_prediction = torch.sum(weak_prediction, dim=1)
        if torch.sum(sum_weak_prediction)!=0:
            max_weak_prediction = torch.argmax(weak_prediction, dim=1)
            strong_prediction = self.encodernet(strong_img)
            unlabeled_loss = F.cross_entropy(strong_prediction, max_weak_prediction, reduction='none')
            unlabeled_loss = torch.mul(unlabeled_loss, sum_weak_prediction)
            unlabeled_loss = torch.sum(unlabeled_loss)/torch.sum(sum_weak_prediction)
            loss = labeled_loss + 0.4*unlabeled_loss
        else:
            loss = labeled_loss
        return loss


encoder = Encoder().to(device)
mainmodel = MainModel(encoder).to(device)
optimizer = optim.Adam(mainmodel.parameters(), lr=1e-3)
history_score = 0

for epoch in range(50):
    encoder.train()
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    index = np.random.permutation(np.arange(len(x_train)))
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]
    max_batch = int(len(x_train) / labeled_batchsize)
    for batch in range(max_batch):
        batch_labeled_img = torch.from_numpy(x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]).float().to(device)
        batch_label = torch.tensor(y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]).to(device)
        batch_unlabeled_img = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_weak_img, batch_strong_img = augment(batch_unlabeled_img)
        batch_weak_img = torch.from_numpy(batch_weak_img).float().to(device)
        batch_strong_img = torch.from_numpy(batch_strong_img).float().to(device)

        optimizer.zero_grad()
        batch_loss = mainmodel(batch_labeled_img, batch_weak_img, batch_strong_img, batch_label)
        batch_loss.backward()
        optimizer.step()


    encoder.eval()
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    max_batch = int(len(x_vad) / batchsize)
    for batch in range(max_batch):
        batch_img = x_vad[batch * batchsize: batch * batchsize + batchsize]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_vad[batch * batchsize:batch * batchsize + batchsize]
        with torch.no_grad():
            batch_pred = encoder(batch_img)
        batch_pred = torch.argmax(batch_pred, dim=1)
        for i in range(batchsize):
            result[batch_label[i]][batch_pred[i]] += 1
    batch_img = x_vad[max_batch * batchsize:]
    batch_img = torch.from_numpy(batch_img).float().to(device)
    batch_label = y_vad[max_batch * batchsize:]
    with torch.no_grad():
        batch_pred = encoder(batch_img)
    batch_pred = torch.argmax(batch_pred, dim=1)
    for i in range(len(x_vad) - max_batch * batchsize):
        result[batch_label[i]][batch_pred[i]] += 1

    acc = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(x_vad)
    se = (result[0][0] + result[1][1] + result[2][2]) / (
            len(x_vad) - result[3][0] - result[3][1] - result[3][2] - result[3][3])
    sp = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
    score = (se + sp) / 2
    if score > history_score:
        history_score = score
        print('vad set: epoch ' + str(epoch) + ':accuracy:' + str(acc) + ';sensitivity:' + str(
            se) + ';specificity:' + str(sp) + ';score:' + str(score))

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
        print('test set: epoch ' + str(epoch) + ':accuracy:' + str(acc) + ';sensitivity:' + str(
            se) + ';specificity:' + str(sp) + ';score:' + str(score))