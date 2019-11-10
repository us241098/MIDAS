import numpy as np
from numpy import loadtxt
from keras.models import load_model
import librosa

def extract_features(file_name):
    ''' Extract mfcc, chroma,mel,contrast and tonnetz features from the audio and return it'''
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs.tolist(),chroma.tolist(),mel.tolist(),contrast.tolist(),tonnetz.tolist()



model = load_model('model_mlp.h5')
model.summary()

import pandas as pd
data = pd.read_csv("Test/test_folder.csv")
print (data.columns)
data_files=data['file_name']
data_label=data['class_label']

for f in data_files:
    data_dim=extract_features('Test/' + f)
    data=np.array([data_dim[0]+data_dim[1]+data_dim[2]+data_dim[3]+data_dim[4]])
    
    k=model.predict(data)
    k=(k == k.max(axis=1, keepdims=1)).astype(float)
    k=k.flatten('C')
    result = np.where(k == 1)
    print (result[0])

    #print label 
    if (result[0]==0):
        print ('disgust')

    if (result[0]==1):
        print ('fear')

    if (result[0]==2):
        print ('happy')

    if (result[0]==3):
        print ('neutral')

    if (result[0]==4):
        print ('sad')


