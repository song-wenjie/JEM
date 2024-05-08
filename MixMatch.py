import glob
import random
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

batchsize = 160
unlabeled_batchsize = 80
labeled_batchsize = 80
print('MixMatch')

AKGC417L = np.load('spectrum_correction/vad'+vad_option+'/AKGC417L.npy')
Meditron = np.load('spectrum_correction/vad'+vad_option+'/Meditron.npy')
Litt3200 = np.load('spectrum_correction/vad'+vad_option+'/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/vad'+vad_option+'/LittC2SE.npy')
HF_Type = np.load('spectrum_correction/vad'+vad_option+'/HF_Type.npy')
Littmann = np.load('spectrum_correction/vad'+vad_option+'/Littmann.npy')
crowdsource = np.load('spectrum_correction/vad'+vad_option+'/crowdsource.npy')


def augment2(img, frequency_masking_para=20, time_masking_para=50):
    img1 = np.empty(shape=(unlabeled_batchsize,3,64,320))
    img2 = np.empty(shape=(unlabeled_batchsize,3,64,320))
    for i in range(unlabeled_batchsize):
        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        temp = np.reshape(temp, newshape=(3,64,320))
        img1[i,:,:,:] = temp

        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        temp = np.reshape(temp, newshape=(3, 64, 320))
        img2[i, :, :, :] = temp
    return img1,img2


def augment1(img, frequency_masking_para=20, time_masking_para=50):
    img1 = np.empty(shape=(labeled_batchsize,3,64,320))
    for i in range(labeled_batchsize):
        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        temp = np.reshape(temp, newshape=(3,64,320))
        img1[i,:,:,:] = temp
    return img1


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


encoder = Encoder().to(device)
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
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
        batch_unlabeled_img = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_unlabeled_img_augment1, batch_unlabeled_img_augment2 = augment2(batch_unlabeled_img)
        batch_unlabeled_img_augment1 = torch.from_numpy(batch_unlabeled_img_augment1).float().to(device)
        batch_unlabeled_img_augment2 = torch.from_numpy(batch_unlabeled_img_augment2).float().to(device)
        batch_unlabeled_prediction1 = encoder(batch_unlabeled_img_augment1)
        batch_unlabeled_prediction1 = F.softmax(batch_unlabeled_prediction1, dim=1)
        batch_unlabeled_prediction2 = encoder(batch_unlabeled_img_augment2)
        batch_unlabeled_prediction2 = F.softmax(batch_unlabeled_prediction2, dim=1)
        batch_unlabeled_prediction = (batch_unlabeled_prediction1 + batch_unlabeled_prediction2) / 2
        batch_unlabeled_prediction = batch_unlabeled_prediction**2
        batch_unlabeled_prediction = batch_unlabeled_prediction/torch.sum(batch_unlabeled_prediction, dim=1, keepdim=True)
        batch_unlabeled_prediction = batch_unlabeled_prediction.detach()

        batch_labeled_img = x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_labeled_img_augment = augment1(batch_labeled_img)
        batch_labeled_img_augment = torch.from_numpy(batch_labeled_img_augment).float().to(device)
        batch_labeled_prediction = y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_labeled_prediction = torch.tensor(batch_labeled_prediction)
        batch_labeled_prediction = F.one_hot(batch_labeled_prediction).float().to(device)

        batch_img = torch.cat((batch_unlabeled_img_augment1, batch_labeled_img_augment), axis=0)
        batch_prediction = torch.cat((batch_unlabeled_prediction, batch_labeled_prediction), axis=0)

        index = np.random.permutation(np.arange(batchsize))
        batch_img = batch_img[index]
        batch_prediction = batch_prediction[index]

        for i in range(labeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r<0.5:
                r = 1-r
            batch_labeled_img_augment[i] = r * batch_labeled_img_augment[i] + (1 - r) * batch_img[i]
            batch_labeled_prediction[i] = r * batch_labeled_prediction[i] + (1 - r) * batch_prediction[i]

        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_img_augment1[i] = r * batch_unlabeled_img_augment1[i] + (1 - r) * batch_img[i + labeled_batchsize]
            batch_unlabeled_prediction[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[i + labeled_batchsize]

        optimizer.zero_grad()
        batch_labeled_img_augment = batch_labeled_img_augment.detach()
        batch_labeled_prediction = batch_labeled_prediction.detach()
        batch_labeled_output = encoder(batch_labeled_img_augment)
        batch_labeled_output = F.log_softmax(batch_labeled_output, dim=1)
        batch_loss1 = -torch.sum(batch_labeled_prediction * batch_labeled_output) / labeled_batchsize

        batch_unlabeled_img_augment1 = batch_unlabeled_img_augment1.detach()
        batch_unlabeled_prediction = batch_unlabeled_prediction.detach()
        batch_unlabeled_output = encoder(batch_unlabeled_img_augment1)
        batch_unlabeled_output = F.softmax(batch_unlabeled_output, dim=1)
        batch_loss2 = F.mse_loss(batch_unlabeled_output, batch_unlabeled_prediction)
        batch_loss = batch_loss1 + batch_loss2
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