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
batchsize = 360
unlabeled_batchsize = 180
labeled_batchsize = 180
print('Tri-Training')


AKGC417L = np.load('spectrum_correction/vad'+vad_option+'/AKGC417L.npy')
Meditron = np.load('spectrum_correction/vad'+vad_option+'/Meditron.npy')
Litt3200 = np.load('spectrum_correction/vad'+vad_option+'/Litt3200.npy')
LittC2SE = np.load('spectrum_correction/vad'+vad_option+'/LittC2SE.npy')
HF_Type = np.load('spectrum_correction/vad'+vad_option+'/HF_Type.npy')
Littmann = np.load('spectrum_correction/vad'+vad_option+'/Littmann.npy')
crowdsource = np.load('spectrum_correction/vad'+vad_option+'/crowdsource.npy')


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
    elif device_index == 't':
        D = D * HF_Type
    elif device_index =='s':
        D = D * Littmann
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
x_vad=x_vad/std


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


encoder0 = Encoder().to(device)
encoder1 = Encoder().to(device)
encoder2 = Encoder().to(device)
optimizer0 = optim.Adam(encoder0.parameters(), lr=1e-3)
optimizer1 = optim.Adam(encoder1.parameters(), lr=1e-3)
optimizer2 = optim.Adam(encoder2.parameters(), lr=1e-3)
history_score = 0

index = np.random.randint(0,len(x_train),(len(x_train),))
x_sample = np.array(x_train)[index]
y_sample = np.array(y_train)[index]
max_batch = int(len(x_sample) / batchsize)
for epoch in range(10):
    index = np.random.permutation(np.arange(len(x_sample)))
    x_sample = np.array(x_sample)[index]
    y_sample = np.array(y_sample)[index]
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_sample[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_sample[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction = encoder0(batch_img)
        optimizer0.zero_grad()
        batch_loss = F.cross_entropy(batch_prediction, batch_label)
        batch_loss.backward()
        optimizer0.step()

index = np.random.randint(0,len(x_train),(len(x_train),))
x_sample = np.array(x_train)[index]
y_sample = np.array(y_train)[index]
max_batch = int(len(x_sample) / batchsize)
for epoch in range(10):
    index = np.random.permutation(np.arange(len(x_sample)))
    x_sample = np.array(x_sample)[index]
    y_sample = np.array(y_sample)[index]
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_sample[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_sample[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction = encoder1(batch_img)
        optimizer1.zero_grad()
        batch_loss = F.cross_entropy(batch_prediction, batch_label)
        batch_loss.backward()
        optimizer1.step()

index = np.random.randint(0,len(x_train),(len(x_train),))
x_sample = np.array(x_train)[index]
y_sample = np.array(y_train)[index]
max_batch = int(len(x_sample) / batchsize)
for epoch in range(10):
    index = np.random.permutation(np.arange(len(x_sample)))
    x_sample = np.array(x_sample)[index]
    y_sample = np.array(y_sample)[index]
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_sample[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_sample[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction = encoder2(batch_img)
        optimizer2.zero_grad()
        batch_loss = F.cross_entropy(batch_prediction, batch_label)
        batch_loss.backward()
        optimizer2.step()

e_prime0 = 0.5
e_prime1 = 0.5
e_prime2 = 0.5
l_prime0 = 0
l_prime1 = 0
l_prime2 = 0
e0 =0
e1 = 0
e2 = 0
update0 = False
update1 = False
update2 = False
improve = True

index = np.random.permutation(np.arange(len(x_train)))
x_train = np.array(x_train)[index]
y_train = np.array(y_train)[index]
index = np.random.permutation(np.arange(len(x_unlabeled)))
x_unlabeled = np.array(x_unlabeled)[index]
while improve:
    encoder0.train()
    encoder1.train()
    encoder2.train()
    update0 = False
    numerator = 0
    denominator = 0
    max_batch = int(len(x_train) / batchsize)
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_train[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_train[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction1 = encoder1(batch_img)
        batch_prediction1 = torch.argmax(batch_prediction1, dim=1)
        batch_prediction2 = encoder2(batch_img)
        batch_prediction2 = torch.argmax(batch_prediction2, dim=1)
        wrong_index = torch.logical_and(batch_prediction1 != batch_label, batch_prediction1 == batch_prediction2)
        numerator = numerator + sum(wrong_index)
        denominator = denominator + sum(batch_prediction1 == batch_prediction2)
    e0 =numerator/denominator
    if e0 < e_prime0:
        x_expand0 = x_unlabeled[0:1]
        y_expand0 = np.array([2])
        max_batch = int(len(x_unlabeled) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_unlabeled[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_prediction1 = encoder1(batch_img)
            batch_prediction1 = torch.argmax(batch_prediction1, dim=1)
            batch_prediction2 = encoder2(batch_img)
            batch_prediction2 = torch.argmax(batch_prediction2, dim=1)
            batch_img = batch_img.detach().cpu().numpy()
            batch_prediction1 = batch_prediction1.cpu().numpy()
            batch_prediction2 = batch_prediction2.cpu().numpy()
            x_expand0 = np.concatenate((x_expand0, batch_img[batch_prediction1 == batch_prediction2]), axis=0)
            y_expand0 = np.concatenate((y_expand0, batch_prediction1[batch_prediction1 == batch_prediction2]), axis=0)
        if l_prime0 == 0:  # no updated before
            l_prime0 = int(e0 / (e_prime0 - e0) + 1)
        if l_prime0 < len(y_expand0):
            if e0 * len(y_expand0) < e_prime0 * l_prime0:
                update0 = True
            elif l_prime0 > e0 / (e_prime0 - e0):
                L_index = np.random.choice(len(y_expand0), int(e_prime0 * l_prime0 / e0 - 1))
                x_expand0, y_expand0 = x_expand0[L_index], y_expand0[L_index]
                update0 = True

    update1 = False
    numerator = 0
    denominator = 0
    max_batch = int(len(x_train) / batchsize)
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_train[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_train[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction0 = encoder0(batch_img)
        batch_prediction0 = torch.argmax(batch_prediction0, dim=1)
        batch_prediction2 = encoder2(batch_img)
        batch_prediction2 = torch.argmax(batch_prediction2, dim=1)
        wrong_index = torch.logical_and(batch_prediction0 != batch_label, batch_prediction0 == batch_prediction2)
        numerator = numerator + sum(wrong_index)
        denominator = denominator + sum(batch_prediction0 == batch_prediction2)
    e1 = numerator / denominator
    if e1 < e_prime1:
        x_expand1 = x_unlabeled[0:1]
        y_expand1 = np.array([2])
        max_batch = int(len(x_unlabeled) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_unlabeled[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_prediction0 = encoder0(batch_img)
            batch_prediction0 = torch.argmax(batch_prediction0, dim=1)
            batch_prediction2 = encoder2(batch_img)
            batch_prediction2 = torch.argmax(batch_prediction2, dim=1)
            batch_img = batch_img.detach().cpu().numpy()
            batch_prediction0 = batch_prediction0.cpu().numpy()
            batch_prediction2 = batch_prediction2.cpu().numpy()
            x_expand1 = np.concatenate((x_expand1, batch_img[batch_prediction0 == batch_prediction2]), axis=0)
            y_expand1 = np.concatenate((y_expand1, batch_prediction0[batch_prediction0 == batch_prediction2]), axis=0)
        if l_prime1 == 0:  # no updated before
            l_prime1 = int(e1 / (e_prime1 - e1) + 1)
        if l_prime1 < len(y_expand1):
            if e1 * len(y_expand1) < e_prime1 * l_prime1:
                update1 = True
            elif l_prime1 > e1 / (e_prime1 - e1):
                L_index = np.random.choice(len(y_expand1), int(e_prime1 * l_prime1 / e1 - 1))
                x_expand1, y_expand1 = x_expand1[L_index], y_expand1[L_index]
                update1 = True


    update2 = False
    numerator = 0
    denominator = 0
    max_batch = int(len(x_train) / batchsize)
    for batch in range(max_batch):
        batch_img = torch.from_numpy(x_train[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
        batch_label = torch.tensor(y_train[batch * batchsize:batch * batchsize + batchsize]).to(device)
        batch_prediction0 = encoder0(batch_img)
        batch_prediction0 = torch.argmax(batch_prediction0, dim=1)
        batch_prediction1 = encoder1(batch_img)
        batch_prediction1 = torch.argmax(batch_prediction1, dim=1)
        wrong_index = torch.logical_and(batch_prediction0 != batch_label, batch_prediction0 == batch_prediction1)
        numerator = numerator + sum(wrong_index)
        denominator = denominator + sum(batch_prediction0 == batch_prediction1)
    e2 = numerator / denominator
    if e2 < e_prime2:
        x_expand2 = x_unlabeled[0:1]
        y_expand2 = np.array([2])
        max_batch = int(len(x_unlabeled) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_unlabeled[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_prediction0 = encoder0(batch_img)
            batch_prediction0 = torch.argmax(batch_prediction0, dim=1)
            batch_prediction1 = encoder1(batch_img)
            batch_prediction1 = torch.argmax(batch_prediction1, dim=1)
            batch_img = batch_img.detach().cpu().numpy()
            batch_prediction0 = batch_prediction0.cpu().numpy()
            batch_prediction1 = batch_prediction1.cpu().numpy()
            x_expand2 = np.concatenate((x_expand2, batch_img[batch_prediction0 == batch_prediction1]), axis=0)
            y_expand2 = np.concatenate((y_expand2, batch_prediction0[batch_prediction0 == batch_prediction1]), axis=0)
        if l_prime2 == 0:  # no updated before
            l_prime2 = int(e2 / (e_prime2 - e2) + 1)
        if l_prime2 < len(y_expand2):
            if e2 * len(y_expand2) < e_prime2 * l_prime2:
                update2 = True
            elif l_prime2 > e2 / (e_prime2 - e2):
                L_index = np.random.choice(len(y_expand2), int(e_prime2 * l_prime2 / e2 - 1))
                x_expand2, y_expand2 = x_expand2[L_index], y_expand2[L_index]
                update2 = True

    if update0:
        e_prime0 = e0
        l_prime0 = len(y_expand0)
        x_expand0 = np.concatenate((x_expand0, x_train), axis=0)
        y_expand0 = np.concatenate((y_expand0, y_train), axis=0)
        index = np.random.permutation(np.arange(len(x_expand0)))
        x_expand0 = np.array(x_expand0)[index]
        y_expand0 = np.array(y_expand0)[index]
        max_batch = int(len(x_expand0) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_expand0[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_label = torch.tensor(y_expand0[batch * batchsize:batch * batchsize + batchsize]).to(device)
            batch_prediction = encoder0(batch_img)
            optimizer0.zero_grad()
            batch_loss = F.cross_entropy(batch_prediction, batch_label)
            batch_loss.backward()
            optimizer0.step()

    if update1:
        e_prime1 = e1
        l_prime1 = len(y_expand1)
        x_expand1 = np.concatenate((x_expand1, x_train), axis=0)
        y_expand1 = np.concatenate((y_expand1, y_train), axis=0)
        index = np.random.permutation(np.arange(len(x_expand1)))
        x_expand1 = np.array(x_expand1)[index]
        y_expand1 = np.array(y_expand1)[index]
        max_batch = int(len(x_expand1) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_expand1[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_label = torch.tensor(y_expand1[batch * batchsize:batch * batchsize + batchsize]).to(device)
            batch_prediction = encoder1(batch_img)
            optimizer1.zero_grad()
            batch_loss = F.cross_entropy(batch_prediction, batch_label)
            batch_loss.backward()
            optimizer1.step()

    if update2:
        e_prime2 = e2
        l_prime2 = len(y_expand2)
        x_expand2 = np.concatenate((x_expand2, x_train), axis=0)
        y_expand2 = np.concatenate((y_expand2, y_train), axis=0)
        index = np.random.permutation(np.arange(len(x_expand2)))
        x_expand2 = np.array(x_expand2)[index]
        y_expand2 = np.array(y_expand2)[index]
        max_batch = int(len(x_expand2) / batchsize)
        for batch in range(max_batch):
            batch_img = torch.from_numpy(x_expand2[batch * batchsize:batch * batchsize + batchsize]).float().to(device)
            batch_label = torch.tensor(y_expand2[batch * batchsize:batch * batchsize + batchsize]).to(device)
            batch_prediction = encoder2(batch_img)
            optimizer2.zero_grad()
            batch_loss = F.cross_entropy(batch_prediction, batch_label)
            batch_loss.backward()
            optimizer2.step()

    encoder0.eval()
    encoder1.eval()
    encoder2.eval()
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    max_batch = int(len(x_vad) / batchsize)
    for batch in range(max_batch):
        vote = np.zeros(shape=(batchsize, 4))
        batch_img = x_vad[batch * batchsize: batch * batchsize + batchsize]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_vad[batch * batchsize:batch * batchsize + batchsize]
        with torch.no_grad():
            pred0 = encoder0(batch_img)
        pred0 = torch.argmax(pred0, dim=1)
        for i in range(batchsize):
            vote[i][pred0[i]] = vote[i][pred0[i]] + 1
        with torch.no_grad():
            pred1 = encoder1(batch_img)
        pred1 = torch.argmax(pred1, dim=1)
        for i in range(batchsize):
            vote[i][pred1[i]] = vote[i][pred1[i]] + 1
        with torch.no_grad():
            pred2 = encoder2(batch_img)
        pred2 = torch.argmax(pred2, dim=1)
        for i in range(batchsize):
            vote[i][pred2[i]] = vote[i][pred2[i]] + 1
        vote = np.argmax(vote, axis=1)
        for i in range(batchsize):
            result[batch_label[i]][vote[i]] += 1

    vote = np.zeros(shape=(len(x_vad)-max_batch*batchsize, 4))
    batch_img = x_vad[max_batch * batchsize:]
    batch_img = torch.from_numpy(batch_img).float().to(device)
    batch_label = y_vad[max_batch * batchsize:]
    with torch.no_grad():
        pred0 = encoder0(batch_img)
    pred0 = torch.argmax(pred0, dim=1)
    for i in range(batchsize):
        vote[i][pred0[i]] = vote[i][pred0[i]] + 1
    with torch.no_grad():
        pred1 = encoder1(batch_img)
    pred1 = torch.argmax(pred1, dim=1)
    for i in range(batchsize):
        vote[i][pred1[i]] = vote[i][pred1[i]] + 1
    with torch.no_grad():
        pred2 = encoder2(batch_img)
    pred2 = torch.argmax(pred2, dim=1)
    for i in range(batchsize):
        vote[i][pred2[i]] = vote[i][pred2[i]] + 1
    vote = np.argmax(vote, axis=1)
    for i in range(batchsize):
        result[batch_label[i]][vote[i]] += 1

    acc = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(x_vad)
    se = (result[0][0] + result[1][1] + result[2][2]) / (
            len(x_vad) - result[3][0] - result[3][1] - result[3][2] - result[3][3])
    sp = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
    score = (se + sp) / 2

    if score>history_score:
        history_score=score
        result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        max_batch = int(len(x_test) / batchsize)
        for batch in range(max_batch):
            vote = np.zeros(shape=(batchsize, 4))
            batch_img = x_test[batch * batchsize: batch * batchsize + batchsize]
            batch_img = torch.from_numpy(batch_img).float().to(device)
            batch_label = y_test[batch * batchsize:batch * batchsize + batchsize]
            with torch.no_grad():
                pred0 = encoder0(batch_img)
            pred0 = torch.argmax(pred0, dim=1)
            for i in range(batchsize):
                vote[i][pred0[i]] = vote[i][pred0[i]] + 1

            with torch.no_grad():
                pred1 = encoder1(batch_img)
            pred1 = torch.argmax(pred1, dim=1)
            for i in range(batchsize):
                vote[i][pred1[i]] = vote[i][pred1[i]] + 1

            with torch.no_grad():
                pred2 = encoder2(batch_img)
            pred2 = torch.argmax(pred2, dim=1)
            for i in range(batchsize):
                vote[i][pred2[i]] = vote[i][pred2[i]] + 1

            vote = np.argmax(vote, axis=1)
            for i in range(batchsize):
                result[batch_label[i]][vote[i]] += 1

        vote = np.zeros(shape=(len(x_test)-max_batch*batchsize, 4))
        batch_img = x_test[max_batch * batchsize:]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_test[max_batch * batchsize:]
        with torch.no_grad():
            pred0 = encoder0(batch_img)
        pred0 = torch.argmax(pred0, dim=1)
        for i in range(batchsize):
            vote[i][pred0[i]] = vote[i][pred0[i]] + 1

        with torch.no_grad():
            pred1 = encoder1(batch_img)
        pred1 = torch.argmax(pred1, dim=1)
        for i in range(batchsize):
            vote[i][pred1[i]] = vote[i][pred1[i]] + 1

        with torch.no_grad():
            pred2 = encoder2(batch_img)
        pred2 = torch.argmax(pred2, dim=1)
        for i in range(batchsize):
            vote[i][pred2[i]] = vote[i][pred2[i]] + 1

        vote = np.argmax(vote, axis=1)
        for i in range(len(x_test)-max_batch*batchsize):
            result[batch_label[i]][vote[i]] += 1

        acc = (result[0][0] + result[1][1] + result[2][2] + result[3][3]) / len(x_test)
        se = (result[0][0] + result[1][1] + result[2][2]) / (
                len(x_test) - result[3][0] - result[3][1] - result[3][2] - result[3][3])
        sp = result[3][3] / (result[3][0] + result[3][1] + result[3][2] + result[3][3])
        score = (se + sp) / 2
        print('epoch ' + str(epoch) + ':accuracy:' + str(acc) + ';sensitivity:' + str(se) + ';specificity:' + str(
            sp) + ';score:' + str(score))

    if update0==False and update1==False and update2==False:
        improve = False