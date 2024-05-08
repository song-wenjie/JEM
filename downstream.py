import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn import svm
import sys

batchsize = 280
vad_option = sys.argv[1]
dataset_type = sys.argv[2]
model_index = sys.argv[3]
device = sys.argv[4]

AKGC417L = np.load('spectrum_correction/vad'+vad_option+'/AKGC417L.npy')
Meditron = np.load('spectrum_correction/vad'+vad_option+'/Meditron.npy')
Litt3200 = np.load('spectrum_correction/vad'+vad_option+'/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/vad'+vad_option+'/LittC2SE.npy')


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
    else:
        D = D * LittC2SE
    D = D.T
    mel = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=64)
    mel = librosa.power_to_db(mel)
    mel = cv2.cvtColor(mel, cv2.COLOR_GRAY2BGR)
    mel = mel.transpose(2, 0, 1)
    return mel


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
elif vad_option == '5':
    test_filenames = filenames5
    vad_filenames = filenames1
    train_filenames = np.concatenate((filenames2, filenames3, filenames4))


# test set
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


# vad set
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
x_vad = x_vad - m
x_vad = x_vad / std


def evaluation_test(pred_test):
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for i in range(len(pred_test)):
        result[y_test[i]][pred_test[i]] += 1
    acc_test = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(pred_test)
    se_test = (result[0][0] + result[1][1] + result[2][2]) / (
            len(pred_test) - result[3][0] - result[3][1] - result[3][2] -
            result[3][3])
    sp_test = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
    score_test = (se_test + sp_test) / 2
    return acc_test, se_test, sp_test, score_test


def evaluation_vad(pred_vad):
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    for i in range(len(pred_vad)):
        result[y_vad[i]][pred_vad[i]] += 1
    acc_vad = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(pred_vad)
    se_vad = (result[0][0] + result[1][1] + result[2][2]) / (
            len(pred_vad) - result[3][0] - result[3][1] - result[3][2] -
            result[3][3])
    sp_vad = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
    score_vad = (se_vad + sp_vad) / 2
    return acc_vad, se_vad, sp_vad, score_vad


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mobileVit = torch.load('model_structure.pt')
        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, img):
        x = self.mobileVit(img)
        x = x.squeeze()
        return x


encoder = Encoder().to(device)
encoder.load_state_dict(torch.load('../models/other_methods/'+dataset_type+'/DAE_encoder'+model_index+'.pth'))
encoder.eval()
max_batch = int(len(x_train)/batchsize)
z_train = np.empty(shape=(len(x_train),960))
for batch in range(max_batch):
    batch_img = torch.from_numpy(x_train[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
    batch_representation = encoder(batch_img)
    batch_representation = batch_representation.detach().cpu().numpy()
    z_train[batch * batchsize:batch * batchsize + batchsize] = batch_representation

batch_img = torch.from_numpy(x_train[-batchsize:]).float().to(device)
batch_representation = encoder(batch_img)
batch_representation = batch_representation.detach().cpu().numpy()
z_train[-batchsize:] = batch_representation

# vad set
historical_score = 0
max_batch = int(len(x_vad)/batchsize)
z_test = np.empty(shape=(len(x_vad),960))
for batch in range(max_batch):
    batch_img = torch.from_numpy(x_vad[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
    batch_representation = encoder(batch_img)
    batch_representation = batch_representation.detach().cpu().numpy()
    z_test[batch * batchsize:batch * batchsize + batchsize] = batch_representation

batch_img = torch.from_numpy(x_vad[-batchsize:]).float().to(device)
batch_representation = encoder(batch_img)
batch_representation = batch_representation.detach().cpu().numpy()
z_test[-batchsize:] = batch_representation
# 网格搜索最佳参数
historical_score = 0
best_c = 0
best_gamma = 0

c_grid = [1, 5,10, 50,100]
gamma_grid = [0.001, 0.01, 0.1, 1, 10, 100]
for c in c_grid:
    for gamma in gamma_grid:
        classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=c, gamma=gamma)
        classifier.fit(z_train, y_train)
        pred = classifier.predict(z_test)
        acc, se, sp, score = evaluation_vad(pred)
        if score>historical_score:
            historical_score = score
            best_c = c
            best_gamma = gamma

#在测试集上进行测试
max_batch = int(len(x_test)/batchsize)
z_test = np.empty(shape=(len(x_test),960))
for batch in range(max_batch):
    batch_img = torch.from_numpy(x_test[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
    batch_representation = encoder(batch_img)
    batch_representation = batch_representation.detach().cpu().numpy()
    z_test[batch * batchsize:batch * batchsize + batchsize] = batch_representation

batch_img = torch.from_numpy(x_test[-batchsize:]).float().to(device)
batch_representation = encoder(batch_img)
batch_representation = batch_representation.detach().cpu().numpy()
z_test[-batchsize:] = batch_representation

classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr', C=best_c, gamma=best_gamma)
classifier.fit(z_train, y_train)
pred = classifier.predict(z_test)

acc, se, sp, score = evaluation_test(pred)
print('accuracy:' + str(acc) + ';sensitivity:' + str(se) + ';specificity:' + str(sp) + ';score:' + str(score) + '\n')