import glob
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import sys


device = sys.argv[1]
dataset_type = sys.argv[2]
iteration = sys.argv[3]
batchsize = 300

HF_Type = np.load('spectrum_correction/official/HF_Type.npy')
Littmann = np.load('spectrum_correction/official/Littmann.npy')
crowdsource = np.load('spectrum_correction/official/crowdsource.npy')

print('Denoise AutoEncoder')

def augment(img, frequency_masking_para=20, time_masking_para=50):
    masked_img = np.empty(shape=(batchsize,3,64,320))
    for i in range(batchsize):
        temp = np.copy(img[i])
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, 64 - f)
        temp[:, f0:f0 + f, :] = 0
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, 320 - t)
        temp[:, :, t0:t0 + t] = 0
        masked_img[i, :, :, :] = temp
    return masked_img


def spectrum(audio,sr, device_index):
    D = np.abs(librosa.stft(audio, n_fft=256, hop_length=64)) ** 2
    while D.shape[1] < 320:
        D = np.concatenate((D, D), axis=1)
    D = D[:, 0:320]
    D = D.T
    if device_index == 's':
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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mobileVit = torch.load('model_structure.pt')
        self.mobileVit.load_state_dict(torch.load('../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt', map_location=device))
        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, img):
        x = self.mobileVit(img)
        x = x.squeeze()
        return x


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
            nn.ReLU(),
        )
        self.fc = nn.Linear(in_features=960, out_features=960*2*10)

    def forward(self, embedding):
        x = self.fc(embedding)
        x = torch.reshape(x, shape=(-1, 960, 2, 10))
        x = self.net(x)
        return x


class MainModel(nn.Module):
    def __init__(self, encodernet, decodernet):
        super(MainModel, self).__init__()
        self.encodernet = encodernet
        self.decodernet = decodernet

    def forward(self, masked_img, original_img):
        representation = self.encodernet(masked_img)
        reconstruction_img = self.decodernet(representation)
        reconstruction_loss = nn.MSELoss()(original_img, reconstruction_img)
        return reconstruction_loss


encoder = Encoder().to(device)
decoder = Decoder().to(device)
mainmodel = MainModel(encoder, decoder).to(device)
optimizer = optim.Adam(mainmodel.parameters(), lr=1e-3)


for epoch in range(101):
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    max_batch = int(len(x_unlabeled) / batchsize)
    for batch in range(max_batch):
        batch_img = x_unlabeled[batch * batchsize:batch * batchsize + batchsize]
        batch_masked_img = augment(batch_img)
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_masked_img = torch.from_numpy(batch_masked_img).float().to(device)
        optimizer.zero_grad()
        batch_loss = mainmodel(batch_masked_img, batch_img)
        batch_loss.backward()
        optimizer.step()
torch.save(encoder.state_dict(), '../models/other_methods/'+dataset_type+'/DAE_encoder'+iteration+'.pth')
torch.save(decoder.state_dict(), '../models/other_methods/'+dataset_type+'/DAE_decoder'+iteration+'.pth')