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
alpha = 10
batchsize = 32
unlabeled_batchsize = 16
labeled_batchsize = 16
print('Variational AutoEncoder')

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
y_train = np.empty(shape=(len(train_filenames),4))
for i in range(len(train_filenames)):
    name = str(train_filenames[i]).split('/')
    if name[-2] == 'crackle':
        y_train[i] = [1, 0, 0, 0]
    elif name[-2] == 'wheeze':
        y_train[i] = [0, 1, 0, 0]
    elif name[-2] == 'both':
        y_train[i] = [0, 0, 1, 0]
    else:
        y_train[i] = [0, 0, 0, 1]
    name = name[-1]
    wav,sr = librosa.load(train_filenames[i],sr=None)
    x_train[i, :, :, :] = spectrum(wav,sr, name[-8])


m = np.mean(x_train, axis=0)
x_train = x_train-m
std = np.std(x_train, axis=0)
x_train = x_train/std
x_test = x_test-m
x_test = x_test/std
x_vad = x_vad-m
x_vad = x_vad/std


class Encoder_y(nn.Module):
    def __init__(self):
        super(Encoder_y, self).__init__()
        self.mobileVit = torch.load('model_structure.pt')
        self.mobileVit.load_state_dict(
            torch.load('../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt', map_location=device))
        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.dense1 = nn.Linear(in_features=960, out_features=960)
        self.dense2 = nn.Linear(in_features=960, out_features=4)

    def forward(self, input):
        embedding = self.mobileVit(input)
        embedding = embedding.squeeze()
        embedding = self.dense1(embedding)
        embedding = F.relu(embedding)
        output = self.dense2(embedding)
        return embedding, output


class Encoder_z(nn.Module):
    def __init__(self):
        super(Encoder_z, self).__init__()
        self.mobileVit = torch.load('model_structure.pt')
        self.mobileVit.load_state_dict(
            torch.load('../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt',
                       map_location=device))
        self.mobileVit.conv_1 = nn.Sequential(
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.Hardswish()
        )

        self.mobileVit.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.dense_mu = nn.Linear(in_features=960, out_features=960)
        self.dense_var = nn.Linear(in_features=960, out_features=960)

    def forward(self, input):
        output = self.mobileVit(input)
        output = output.squeeze()
        output_mu = self.dense_mu(output)
        output_var = self.dense_var(output)
        output_var = torch.exp(output_var)
        return output_mu, output_var


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

    def forward(self, input):
        x = self.fc(input)
        x = torch.reshape(x, shape=(-1, 960, 2, 10))
        output = self.net(x)
        return output


def log_Gaussian_pro(x,mean,var):
    d=x.size(1)
    return -d/2*np.log(2*np.pi)-1/2*torch.sum(torch.log(var+1e-10),1)-1/2*torch.sum((x-mean)**2/var,1)


class M2(nn.Module):
    def __init__(self, encoder_y_net, encoder_z_net, decoder_net, alpha):
        super(M2, self).__init__()
        self.encoder_y_net = encoder_y_net
        self.encoder_z_net = encoder_z_net
        self.decoder_net = decoder_net
        self.alpha = alpha

    def forward(self, lx,ly,lxy,ux):
        # labeled data
        z_mu, z_var = self.encoder_z_net(lxy)  # bl * 50
        z = torch.randn_like(z_mu) * torch.sqrt(z_var) + z_mu
        log_q_z = log_Gaussian_pro(z, z_mu, z_var)  # bl
        x = self.decoder_net(torch.cat([z, ly], 1))  # bl * 784
        flatten_lx = lx.view(-1, 3*64*320)
        flatten_x = x.view(-1,3*64*320)
        log_p_x = -torch.sum(F.binary_cross_entropy_with_logits(flatten_x, flatten_lx, reduction='none'), 1)  # bl
        log_p_y = np.log(0.1)
        log_p_z = log_Gaussian_pro(z, torch.zeros_like(z), torch.ones_like(z))  # bl
        Labled_loss = torch.mean(log_q_z - log_p_x - log_p_y - log_p_z)

        # supervised learning
        _, y_ = self.encoder_y_net(lx)  # bl * 10
        Sup_loss = self.alpha * F.cross_entropy(y_, torch.argmax(ly, 1), reduction='mean')

        # unlabeled data
        _, uq_y = self.encoder_y_net(ux)
        uq_y = F.softmax(uq_y, dim=-1)  # bu *10

        Unlabled_loss = 0

        for i in range(4):
            uy = torch.ones(size=(unlabeled_batchsize,3,64,320)).to(device)
            uy = uy*i
            uz_mu, uz_var = self.encoder_z_net(torch.cat([ux, uy], 1))  # bu * 50
            uz = torch.randn_like(uz_mu) * torch.sqrt(uz_var) + uz_mu  # bu * 50
            ulog_q_z = log_Gaussian_pro(uz, uz_mu, uz_var)  # bu

            uy_ = torch.zeros(size=(unlabeled_batchsize,4)).to(device)
            uy_[:, i] = 1
            xx = self.decoder_net(torch.cat([uz, uy_], 1))  # bu * 784
            flatten_ux = ux.view(-1, 3*64*320)
            flatten_xx = xx.view(-1, 3*64*320)
            ulog_p_x = -torch.sum(F.binary_cross_entropy_with_logits(flatten_xx, flatten_ux, reduction='none'), 1)  # bu
            ulog_p_y = np.log(0.1)
            ulog_p_z = log_Gaussian_pro(uz, torch.zeros_like(uz), torch.ones_like(uz))  # bu
            Unlabled_loss = Unlabled_loss + (ulog_q_z - ulog_p_x - ulog_p_y - ulog_p_z) * uq_y[:, i] + uq_y[:,i] * torch.log(uq_y[:, i] + 1e-10)
        Unlabled_loss = torch.mean(Unlabled_loss)
        return Labled_loss, Unlabled_loss, Sup_loss


encoder_y = Encoder_y().to(device)
encoder_z = Encoder_z().to(device)
decoder = Decoder().to(device)
m2 = M2(encoder_y, encoder_z, decoder, alpha).to(device)
optimizer = optim.Adam(m2.parameters(), lr=2e-4, betas=(0.5, 0.9))
history_score = 0


for epoch in range(50):
    encoder_y.train()
    encoder_z.train()
    decoder.train()
    index = np.random.permutation(np.arange(len(x_unlabeled)))
    x_unlabeled = np.array(x_unlabeled)[index]
    index = np.random.permutation(np.arange(len(x_train)))
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]
    max_batch = int(len(x_train) / labeled_batchsize)
    for batch in range(max_batch):
        batch_lx = x_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_lx = torch.from_numpy(batch_lx).float().to(device)
        batch_ux = x_unlabeled[batch * unlabeled_batchsize:batch * unlabeled_batchsize + unlabeled_batchsize]
        batch_ux = torch.from_numpy(batch_ux).float().to(device)
        batch_ly = y_train[batch * labeled_batchsize:batch * labeled_batchsize + labeled_batchsize]
        batch_ly = torch.from_numpy(batch_ly).float().to(device)

        batch_lxy = torch.ones_like(batch_lx)
        for i in range(labeled_batchsize):
            batch_lxy[i] = batch_lxy[i]*torch.argmax(batch_ly[i])
        batch_lxy = torch.cat((batch_lx, batch_lxy),dim=1)
        optimizer.zero_grad()
        batch_labeled_loss, batch_unlabeled_loss, batch_sup_loss =m2(batch_lx, batch_ly, batch_lxy, batch_ux)
        batch_loss = batch_labeled_loss + batch_unlabeled_loss + batch_sup_loss
        batch_loss.backward()
        optimizer.step()

    encoder_y.eval()
    result = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    max_batch = int(len(x_vad) / batchsize)
    for batch in range(max_batch):
        batch_img = x_vad[batch * batchsize: batch * batchsize + batchsize]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_vad[batch * batchsize:batch * batchsize + batchsize]
        with torch.no_grad():
            _, batch_pred = encoder_y(batch_img)
        batch_pred = torch.argmax(batch_pred, dim=1)
        for i in range(batchsize):
            result[batch_label[i]][batch_pred[i]] += 1
    batch_img = x_vad[max_batch * batchsize:]
    batch_img = torch.from_numpy(batch_img).float().to(device)
    batch_label = y_vad[max_batch * batchsize:]
    with torch.no_grad():
        _, batch_pred = encoder_y(batch_img)
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
                _, batch_pred = encoder_y(batch_img)
            batch_pred = torch.argmax(batch_pred, dim=1)
            for i in range(batchsize):
                result[batch_label[i]][batch_pred[i]] += 1
        batch_img = x_test[max_batch * batchsize:]
        batch_img = torch.from_numpy(batch_img).float().to(device)
        batch_label = y_test[max_batch * batchsize:]
        with torch.no_grad():
            _, batch_pred = encoder_y(batch_img)
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