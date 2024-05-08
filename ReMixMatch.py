import glob
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import cv2
import sys

device = sys.argv[1]
dataset_type = sys.argv[2]
vad_option = sys.argv[3]
batchsize = 56
unlabeled_batchsize = 28
labeled_batchsize = 28
print('ReMixMatch')

AKGC417L = np.load('spectrum_correction/vad'+vad_option+'/AKGC417L.npy')
Meditron = np.load('spectrum_correction/vad'+vad_option+'/Meditron.npy')
Litt3200 = np.load('spectrum_correction/vad'+vad_option+'/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/vad'+vad_option+'/LittC2SE.npy')
HF_Type = np.load('spectrum_correction/vad'+vad_option+'/HF_Type.npy')
Littmann = np.load('spectrum_correction/vad'+vad_option+'/Littmann.npy')
crowdsource = np.load('spectrum_correction/vad'+vad_option+'/crowdsource.npy')


def weak_augmentation(img, weak_frequency_masking_para=10, weak_time_masking_para=20):
    augmented_img = np.empty(shape=(unlabeled_batchsize, 3, 64, 320))
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
        augmented_img[i, :, :, :] = temp
    return augmented_img


def strong_augmentation(img, strong_frequency_masking_para=20, strong_time_masking_para=50):
    augmented_img = np.empty(shape=(unlabeled_batchsize, 3, 64, 320))
    for i in range(unlabeled_batchsize):
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
        augmented_img[i, :, :, :] = temp
    return augmented_img


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


index = np.random.permutation(np.arange(len(train_filenames)))
x_train = np.empty(shape=(len(train_filenames),3,64,320))
y_train = np.empty(shape=(len(train_filenames),4))
for i in range(len(train_filenames)):
    name = str(train_filenames[i]).split('/')
    if name[-2] == 'crackle':
        y_train[i,:] = [1,0,0,0]
    elif name[-2] == 'wheeze':
        y_train[i,:] = [0,1,0,0]
    elif name[-2] == 'both':
        y_train[i,:] = [0,0,1,0]
    else:
        y_train[i,:] = [0,0,0,1]
    name = name[-1]
    wav,sr = librosa.load(train_filenames[i],sr=None)
    x_train[i, :, :, :] = spectrum(wav, sr, name[-8])


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
if vad_option=='1':
    labeled_class_distribution = np.array([883,604,355,2024]) / 3866
elif vad_option=='2':
    labeled_class_distribution = np.array([1221,645,391,2238]) / 4495
elif vad_option=='3':
    labeled_class_distribution = np.array([1059,457,228,1999]) / 3743
elif vad_option=='4':
    labeled_class_distribution = np.array([1330,438,234,2271]) / 4273
elif vad_option=='5':
    labeled_class_distribution = np.array([1099,514,310,2394]) / 4317

buffer = collections.deque(maxlen=128)

for epoch in range(50):
    encoder.train()
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    index = np.random.permutation(np.arange(len(x_train)))
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]
    max_batch = int(len(x_train) / labeled_batchsize)
    for batch in range(max_batch):
        batch_labeled_img = x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_labeled_strong_augmented_img = strong_augmentation(batch_labeled_img)  # 有标注样本1次强augmentation
        batch_labeled_prediction = y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]

        batch_unlabeled_img = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_unlabeled_strong_augmented_img1 = strong_augmentation(batch_unlabeled_img)  # 无标注样本8次强augmetation
        batch_unlabeled_strong_augmented_img2 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img3 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img4 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img5 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img6 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img7 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img8 = strong_augmentation(batch_unlabeled_img)
        batch_unlabeled_strong_augmented_img = np.copy(batch_unlabeled_strong_augmented_img1)
        batch_unlabeled_weak_augmented_img = weak_augmentation(batch_unlabeled_img)  # 无标注样本1次弱augmentation

        # 弱扩充的无标注样本送入编码器得到模型预测值
        batch_unlabeled_weak_augmented_img = torch.from_numpy(batch_unlabeled_weak_augmented_img).float().to(device)
        batch_unlabeled_prediction = encoder(batch_unlabeled_weak_augmented_img)
        batch_unlabeled_prediction = F.softmax(batch_unlabeled_prediction, dim=-1)

        # 得到无标注样本的类别分布
        batch_unlabeled_weak_augmented_img = batch_unlabeled_weak_augmented_img.cpu().numpy()
        batch_unlabeled_prediction = batch_unlabeled_prediction.detach().cpu().numpy()
        batch_unlabeled_prediction_original = np.copy(batch_unlabeled_prediction)
        batch_unlabeled_prediction_original = np.mean(batch_unlabeled_prediction_original, axis=0)
        buffer.append(batch_unlabeled_prediction_original)

        unlabeled_class_distribution = np.array([0,0,0,0])
        for b in range(len(buffer)):
            unlabeled_class_distribution = unlabeled_class_distribution + buffer[b]
        unlabeled_class_distribution = unlabeled_class_distribution/len(buffer)

        batch_unlabeled_prediction = (batch_unlabeled_prediction * labeled_class_distribution)/unlabeled_class_distribution
        batch_unlabeled_prediction = batch_unlabeled_prediction / np.sum(batch_unlabeled_prediction, axis=-1, keepdims=True)

        batch_unlabeled_prediction = batch_unlabeled_prediction**2
        batch_unlabeled_prediction = batch_unlabeled_prediction / np.sum(batch_unlabeled_prediction, axis=-1, keepdims=True)

        batch_img = np.concatenate((batch_labeled_strong_augmented_img, batch_unlabeled_weak_augmented_img,
                               batch_unlabeled_strong_augmented_img1, batch_unlabeled_strong_augmented_img2,
                               batch_unlabeled_strong_augmented_img3, batch_unlabeled_strong_augmented_img4,
                               batch_unlabeled_strong_augmented_img5, batch_unlabeled_strong_augmented_img6,
                               batch_unlabeled_strong_augmented_img7, batch_unlabeled_strong_augmented_img8), axis=0)
        batch_prediction = np.concatenate((batch_labeled_prediction, batch_unlabeled_prediction, batch_unlabeled_prediction,
                                      batch_unlabeled_prediction, batch_unlabeled_prediction,
                                      batch_unlabeled_prediction, batch_unlabeled_prediction,
                                      batch_unlabeled_prediction, batch_unlabeled_prediction,
                                      batch_unlabeled_prediction), axis=0)
        # shuffle
        index = np.random.permutation(np.arange(unlabeled_batchsize*10))
        batch_img = np.array(batch_img)[index]
        batch_prediction = np.array(batch_prediction)[index]
        # mixup
        for i in range(labeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r<0.5:
                r = 1-r
            batch_labeled_strong_augmented_img[i] = r * batch_labeled_strong_augmented_img[i] + (1 - r) * batch_img[i]
            batch_labeled_prediction[i] = r * batch_labeled_prediction[i] + (1 - r) * batch_prediction[i]

        batch_unlabeled_weak_prediction = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_weak_augmented_img[i] = r * batch_unlabeled_weak_augmented_img[i] + (1 - r) * batch_img[i + labeled_batchsize]
            batch_unlabeled_weak_prediction[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[i + labeled_batchsize]

        batch_unlabeled_strong_prediction1 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img1[i] = r * batch_unlabeled_strong_augmented_img1[i] + (1 - r) * batch_img[
                i + labeled_batchsize + unlabeled_batchsize]
            batch_unlabeled_strong_prediction1[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize]

        batch_unlabeled_strong_prediction2 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img2[i] = r * batch_unlabeled_strong_augmented_img2[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 2]
            batch_unlabeled_strong_prediction2[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 2]

        batch_unlabeled_strong_prediction3 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img3[i] = r * batch_unlabeled_strong_augmented_img3[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 3]
            batch_unlabeled_strong_prediction3[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 3]

        batch_unlabeled_strong_prediction4 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img4[i] = r * batch_unlabeled_strong_augmented_img4[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 4]
            batch_unlabeled_strong_prediction4[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 4]

        batch_unlabeled_strong_prediction5 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img5[i] = r * batch_unlabeled_strong_augmented_img5[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 5]
            batch_unlabeled_strong_prediction5[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 5]

        batch_unlabeled_strong_prediction6 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img6[i] = r * batch_unlabeled_strong_augmented_img6[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 6]
            batch_unlabeled_strong_prediction6[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 6]

        batch_unlabeled_strong_prediction7 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img7[i] = r * batch_unlabeled_strong_augmented_img7[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 7]
            batch_unlabeled_strong_prediction7[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 7]

        batch_unlabeled_strong_prediction8 = np.empty(shape=(unlabeled_batchsize, 4))
        for i in range(unlabeled_batchsize):
            r = np.random.beta(0.75, 0.75)
            if r < 0.5:
                r = 1 - r
            batch_unlabeled_strong_augmented_img8[i] = r * batch_unlabeled_strong_augmented_img8[i] + (1 - r) * \
                                                       batch_img[i + labeled_batchsize + unlabeled_batchsize * 8]
            batch_unlabeled_strong_prediction8[i] = r * batch_unlabeled_prediction[i] + (1 - r) * batch_prediction[
                i + labeled_batchsize + unlabeled_batchsize * 8]

        optimizer.zero_grad()
        batch_labeled_strong_augmented_img = torch.from_numpy(batch_labeled_strong_augmented_img).float().to(device)
        batch_labeled_prediction = torch.from_numpy(batch_labeled_prediction).float().to(device)
        batch_labeled_output = encoder(batch_labeled_strong_augmented_img)
        batch_labeled_output = F.log_softmax(batch_labeled_output, dim=-1)
        batch_labeled_loss = -torch.sum(batch_labeled_prediction * batch_labeled_output)/labeled_batchsize

        batch_unlabeled_strong_augmented_img1 = torch.from_numpy(batch_unlabeled_strong_augmented_img1).float().to(device)
        batch_unlabeled_strong_prediction1 = torch.from_numpy(batch_unlabeled_strong_prediction1).float().to(device)
        batch_unlabeled_strong_output1 = encoder(batch_unlabeled_strong_augmented_img1)
        batch_unlabeled_strong_output1 = F.log_softmax(batch_unlabeled_strong_output1, dim=-1)
        batch_unlabeled_strong_loss1 = -torch.sum(batch_unlabeled_strong_prediction1 * batch_unlabeled_strong_output1)/unlabeled_batchsize

        batch_unlabeled_strong_augmented_img2 = torch.from_numpy(batch_unlabeled_strong_augmented_img2).float().to(device)
        batch_unlabeled_strong_prediction2 = torch.from_numpy(batch_unlabeled_strong_prediction2).float().to(device)
        batch_unlabeled_strong_output2 = encoder(batch_unlabeled_strong_augmented_img2)
        batch_unlabeled_strong_output2 = F.log_softmax(batch_unlabeled_strong_output2, dim=-1)
        batch_unlabeled_strong_loss2 = -torch.sum(batch_unlabeled_strong_prediction2 * batch_unlabeled_strong_output2) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img3 = torch.from_numpy(batch_unlabeled_strong_augmented_img3).float().to(device)
        batch_unlabeled_strong_prediction3 = torch.from_numpy(batch_unlabeled_strong_prediction3).float().to(device)
        batch_unlabeled_strong_output3 = encoder(batch_unlabeled_strong_augmented_img3)
        batch_unlabeled_strong_output3 = F.log_softmax(batch_unlabeled_strong_output3, dim=-1)
        batch_unlabeled_strong_loss3 = -torch.sum(batch_unlabeled_strong_prediction3 * batch_unlabeled_strong_output3) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img4 = torch.from_numpy(batch_unlabeled_strong_augmented_img4).float().to(device)
        batch_unlabeled_strong_prediction4 = torch.from_numpy(batch_unlabeled_strong_prediction4).float().to(device)
        batch_unlabeled_strong_output4 = encoder(batch_unlabeled_strong_augmented_img4)
        batch_unlabeled_strong_output4 = F.log_softmax(batch_unlabeled_strong_output4, dim=-1)
        batch_unlabeled_strong_loss4 = -torch.sum(batch_unlabeled_strong_prediction4 * batch_unlabeled_strong_output4) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img5 = torch.from_numpy(batch_unlabeled_strong_augmented_img5).float().to(device)
        batch_unlabeled_strong_prediction5 = torch.from_numpy(batch_unlabeled_strong_prediction5).float().to(device)
        batch_unlabeled_strong_output5 = encoder(batch_unlabeled_strong_augmented_img5)
        batch_unlabeled_strong_output5 = F.log_softmax(batch_unlabeled_strong_output5, dim=-1)
        batch_unlabeled_strong_loss5 = -torch.sum(batch_unlabeled_strong_prediction5 * batch_unlabeled_strong_output5) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img6 = torch.from_numpy(batch_unlabeled_strong_augmented_img6).float().to(device)
        batch_unlabeled_strong_prediction6 = torch.from_numpy(batch_unlabeled_strong_prediction6).float().to(device)
        batch_unlabeled_strong_output6 = encoder(batch_unlabeled_strong_augmented_img6)
        batch_unlabeled_strong_output6 = F.log_softmax(batch_unlabeled_strong_output6, dim=-1)
        batch_unlabeled_strong_loss6 = -torch.sum(batch_unlabeled_strong_prediction6 * batch_unlabeled_strong_output6) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img7 = torch.from_numpy(batch_unlabeled_strong_augmented_img7).float().to(device)
        batch_unlabeled_strong_prediction7 = torch.from_numpy(batch_unlabeled_strong_prediction7).float().to(device)
        batch_unlabeled_strong_output7 = encoder(batch_unlabeled_strong_augmented_img7)
        batch_unlabeled_strong_output7 = F.log_softmax(batch_unlabeled_strong_output7, dim=-1)
        batch_unlabeled_strong_loss7 = -torch.sum(batch_unlabeled_strong_prediction7 * batch_unlabeled_strong_output7) / unlabeled_batchsize

        batch_unlabeled_strong_augmented_img8 = torch.from_numpy(batch_unlabeled_strong_augmented_img8).float().to(device)
        batch_unlabeled_strong_prediction8 = torch.from_numpy(batch_unlabeled_strong_prediction8).float().to(device)
        batch_unlabeled_strong_output8 = encoder(batch_unlabeled_strong_augmented_img8)
        batch_unlabeled_strong_output8 = F.log_softmax(batch_unlabeled_strong_output8, dim=-1)
        batch_unlabeled_strong_loss8 = -torch.sum(batch_unlabeled_strong_prediction8 * batch_unlabeled_strong_output8) / unlabeled_batchsize

        batch_unlabeled_weak_augmented_img = torch.from_numpy(batch_unlabeled_weak_augmented_img).float().to(device)
        batch_unlabeled_weak_prediction = torch.from_numpy(batch_unlabeled_weak_prediction).float().to(device)
        batch_unlabeled_weak_output = encoder(batch_unlabeled_weak_augmented_img)
        batch_unlabeled_weak_output = F.log_softmax(batch_unlabeled_weak_output, dim=-1)
        batch_unlabeled_weak_loss = -torch.sum(batch_unlabeled_weak_prediction*batch_unlabeled_weak_output)/unlabeled_batchsize

        batch_unlabeled_strong_augmented_img = torch.from_numpy(batch_unlabeled_strong_augmented_img).float().to(device)
        batch_unlabeled_prediction = torch.from_numpy(batch_unlabeled_prediction).float().to(device)
        batch_unlabeled_output = encoder(batch_unlabeled_strong_augmented_img)
        batch_unlabeled_output = F.log_softmax(batch_unlabeled_output, dim=-1)
        batch_unlabeled_loss = -torch.sum(batch_unlabeled_prediction * batch_unlabeled_output) / unlabeled_batchsize

        batch_loss = batch_labeled_loss + (1 / 12) * (
                batch_unlabeled_weak_loss + batch_unlabeled_strong_loss1 + batch_unlabeled_strong_loss2 + batch_unlabeled_strong_loss3 +
                batch_unlabeled_strong_loss4 + batch_unlabeled_strong_loss5 + batch_unlabeled_strong_loss6 + batch_unlabeled_strong_loss7 +
                batch_unlabeled_strong_loss8) + 0.25 * batch_unlabeled_loss
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