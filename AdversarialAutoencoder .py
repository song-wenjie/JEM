import glob
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import random
import sys

device = sys.argv[1]
dataset_type = sys.argv[2]
vad_option = sys.argv[3]
batchsize = 240
unlabeled_batchsize = 120
labeled_batchsize = 120
alpha = 10
print('Adversarial AutoEncoder')

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
        self.mobileVit.load_state_dict(
            torch.load('../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt', map_location=device))
        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.z_map = nn.Linear(in_features=960, out_features=960)  #决定了隐变量的维度
        self.y_map = nn.Sequential(
            nn.Linear(in_features=960, out_features=4),
            nn.Softmax()
        )

    def forward(self, input):
        output = self.mobileVit(input) # batchsize * 960 * 1* 1
        output = output.squeeze()  # batchsize * 960
        output_z = self.z_map(output)  # batchsize * 960
        output_y = self.y_map(output) # batchsize * 4
        return output_z, output_y


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=960, out_channels=320, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=320, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(3),
        )
        self.fc = nn.Linear(in_features=960+4, out_features=960 * 2 * 10)

    def forward(self, input_z, input_y):
        x = torch.concatenate((input_z, input_y), dim=-1)
        x = self.fc(x)
        x = torch.reshape(x, shape=(-1, 960, 2, 10))
        output = self.net(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=960+4, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input_z, input_y):
        x = torch.concatenate((input_z, input_y), dim=-1)
        x = self.net(x)
        return x


encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)
optimizerEncoder = optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerDecoder = optim.Adam(decoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerDiscriminator = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
history_score = 0
real_label = 1.0
fack_label = 0.0
TINY = 1e-8

for epoch in range(50):
    encoder.train()
    decoder.train()
    discriminator.train()
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    index = np.random.permutation(np.arange(len(x_train)))
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]
    max_batch = int(len(x_train) / labeled_batchsize)
    for batch in range(max_batch):
        batch_labeled_img = x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_labeled_img = torch.from_numpy(batch_labeled_img).float().to(device)
        batch_unlabeled_img = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_unlabeled_img = torch.from_numpy(batch_unlabeled_img).float().to(device)
        batch_img = torch.concatenate((batch_labeled_img, batch_unlabeled_img))  # labeled + unlabeled
        batch_label = y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_label = torch.tensor(batch_label).to(device)

        # discriminator固定，用重构误差更新encoder和decoder
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True
        for p in discriminator.parameters():
            p.requires_grad = False
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerDecoder.zero_grad()
        optimizerDiscriminator.zero_grad()

        z_output, y_output = encoder(batch_img)
        reconstructed_img = decoder(z_output, y_output)
        recon_loss = F.mse_loss(reconstructed_img, batch_img)   # recon_loss = 2
        recon_loss.backward()
        optimizerEncoder.step()
        optimizerDecoder.step()

        # encoder和decoder固定，用分类损失更新discriminator
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerDecoder.zero_grad()
        optimizerDiscriminator.zero_grad()

        z_real_gauss = Variable(torch.randn(size=(batchsize, 960))).to(device)
        y_real_gauss = np.empty(shape=(batchsize,4))
        for i in range(batchsize):
            r = random.random()
            if r<0.2933:
                y_real_gauss[i] = [1, 0, 0, 0]
            elif r>0.2933 and r<0.4143:
                y_real_gauss[i] = [0, 1, 0, 0]
            elif r>0.4143 and r<0.5019:
                y_real_gauss[i] = [0, 0, 1, 0]
            else:
                y_real_gauss[i] = [0, 0, 0, 1]
        y_real_gauss = torch.from_numpy(y_real_gauss).float().to(device)

        z_fack_gauss, y_fack_gauss = encoder(batch_img)

        D_real_gauss = discriminator(z_real_gauss, y_real_gauss)
        D_fack_gauss = discriminator(z_fack_gauss, y_fack_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss+TINY)+torch.log(1-D_fack_gauss+TINY)) # 1.4
        D_loss.backward()
        optimizerDiscriminator.step()

        # decoder和discriminator固定，用梯度反向传播更新encoder
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = False
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerDecoder.zero_grad()
        optimizerDiscriminator.zero_grad()

        z_fack_gauss, y_fack_gauss = encoder(batch_img)
        D_fack_gauss = discriminator(z_fack_gauss, y_fack_gauss)
        D_loss = -torch.mean(torch.log(D_fack_gauss+TINY)) # 0.7
        D_loss.backward()
        optimizerEncoder.step()

        # 固定decoder和discriminator,用多类别交叉熵损失更新encoder
        for p in encoder.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = False
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()
        optimizerEncoder.zero_grad()
        optimizerDecoder.zero_grad()
        optimizerDiscriminator.zero_grad()

        _, y_pred = encoder(batch_labeled_img)
        ce_loss = alpha * F.nll_loss(torch.log(y_pred+TINY), batch_label) # 1.4
        ce_loss.backward()
        optimizerEncoder.step()

    encoder.eval()
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    max_batch = int(len(x_vad) / batchsize)
    for batch in range(max_batch):
        batch_img = x_vad[batch * batchsize: batch * batchsize + batchsize]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_vad[batch * batchsize:batch * batchsize + batchsize]
        with torch.no_grad():
            _, batch_pred = encoder(batch_img)
        batch_pred = torch.argmax(batch_pred, dim=1)
        for i in range(batchsize):
            result[batch_label[i]][batch_pred[i]] += 1
    batch_img = x_vad[max_batch * batchsize:]
    batch_img = torch.from_numpy(batch_img).float().to(device)
    batch_label = y_vad[max_batch * batchsize:]
    with torch.no_grad():
        _, batch_pred = encoder(batch_img)
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
                _, batch_pred = encoder(batch_img)
            batch_pred = torch.argmax(batch_pred, dim=1)
            for i in range(batchsize):
                result[batch_label[i]][batch_pred[i]] += 1
        batch_img = x_test[max_batch * batchsize:]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_test[max_batch * batchsize:]
        with torch.no_grad():
            _, batch_pred = encoder(batch_img)
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