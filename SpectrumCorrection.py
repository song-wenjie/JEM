import numpy as np
import librosa
import glob


filenames = glob.glob('dataset/official/train/*/*')

AKGC417L = []
Meditron = []
Litt3200 = []
LittC2SE = []

for i in range(len(filenames)):
    wav, sr = librosa.load(filenames[i], sr=None)
    D = np.abs(librosa.stft(wav, n_fft=256, hop_length=64)) ** 2
    D = np.mean(D, axis=1)
    name = str(filenames[i]).split('/')
    name = name[-1]
    if name[-8] == 'L':
        AKGC417L.append(D)
    elif name[-8]=='n':
        Meditron.append(D)
    elif name[-8] == '0':
        Litt3200.append(D)
    else:
        LittC2SE.append(D)

AKGC417L = np.array(AKGC417L)
Meditron = np.array(Meditron)
Litt3200 = np.array(Litt3200)
LittC2SE = np.array(LittC2SE)

AKGC417L = np.mean(AKGC417L, axis=0)
Meditron = np.mean(Meditron, axis=0)
Litt3200 = np.mean(Litt3200, axis=0)
LittC2SE = np.mean(LittC2SE, axis=0)

reference = (AKGC417L + Meditron + Litt3200 + LittC2SE)/4
AKGC417L = reference/AKGC417L
Meditron = reference/Meditron
Litt3200 = reference/Litt3200
LittC2SE = reference/LittC2SE

np.save('spectrum_correction/official/AKGC417L.npy', AKGC417L)
np.save('spectrum_correction/official/Meditron.npy', Meditron)
np.save('spectrum_correction/official/Litt3200.npy', Litt3200)
np.save('spectrum_correction/official/LittC2SE.npy', LittC2SE)



filenames = glob.glob('dataset/HF_Lung_V1/*')

Littmann = []
HF_Type = []

for i in range(len(filenames)):
    wav, sr = librosa.load(filenames[i], sr=None)
    D = np.abs(librosa.stft(wav, n_fft=256, hop_length=64)) ** 2
    D = np.mean(D, axis=1)
    name = str(filenames[i]).split('/')
    name = name[-1]
    if name[0] == 's':
        Littmann.append(D)
    else:
        HF_Type.append(D)

Littmann = np.array(Littmann)
HF_Type = np.array(HF_Type)

Littmann = np.mean(Littmann, axis=0)
HF_Type = np.mean(HF_Type, axis=0)

Littmann = reference/Littmann
HF_Type = reference/HF_Type

np.save('spectrum_correction/official/HF_Type.npy', HF_Type)
np.save('spectrum_correction/official/Littmann.npy', Littmann)



filenames = glob.glob('dataset/covid/*')
crowdsource = []

for i in range(len(filenames)):
    wav, sr = librosa.load(filenames[i], sr=None)
    D = np.abs(librosa.stft(wav, n_fft=256, hop_length=64)) ** 2
    D = np.mean(D, axis=1)
    crowdsource.append(D)

crowdsource = np.array(crowdsource)
crowdsource = np.mean(crowdsource, axis=0)
crowdsource = reference/crowdsource
np.save('spectrum_correction/official/crowdsource.npy', crowdsource)