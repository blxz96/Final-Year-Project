import streamlit as st
import os
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import pandas as pd
import numpy as np
import torch.nn.functional as F
import random
import scipy
import math
from scipy import signal
import librosa.display
import torch.nn as nn
from torch.utils.data import DataLoader
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import seaborn as sns
from matplotlib import colors

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("");
        background-size: 'auto'
    }

    </style>
    """,
    unsafe_allow_html=True
)
    
GAMMA = 3.33
DATA = 'emodb_MFCC'
LAYERS = 1
ROOT = './Dataset/{}'.format(DATA)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_PATH = './model/{}_DANN_{}L-CNN-GRU_DAP{}_CV'.format(DATA,LAYERS,GAMMA)

from modelArchitecture import EmoDBDataset2, FeatureExtractor, EmotionClassifier, SpeakerClassifier

st.markdown("<h1 style='text-align: center; font-family:georgia;'>Speaker Invariant Emotion Recognition with Adversarial Learning</h1>", unsafe_allow_html= True)

#st.beta_container()
col1, col2, col3, col4= st.beta_columns([2,2,2,0.5])

with col1:
    st.markdown("<h3 style='text-align: left;font-family:courier new;'>The model shown in this presentation is trained with 5-fold leave-two-speakers-out cross-validation on Emo-DB dataset</h3>",unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;font-family:courier new,garamond;'>Speakers' Profile(ID, Gender, Age):</h3>",unsafe_allow_html=True)

    speaker_infos = pd.DataFrame({
      'Speaker ID': [3,8,9,10,11,12,13,14,15,16],
      'Gender':['Male','Female','Female','Male','Male','Male','Female','Female','Male','Female'],
      'Age': [31,34,21,32,26,30,32,35,25,31]  
    }).T
    speaker_infos = speaker_infos.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen','font-size': '12pt','text-align' : 'center',
                           'border-color': 'white'})
    st.write(speaker_infos)

    st.markdown("<h3 style='text-align: center;font-family:courier new,garamond;'>Settings of the 5-Fold Cross-Validation:</h3>",unsafe_allow_html=True)


    fold_infos = pd.DataFrame({
      'Training Speakers': ["12,13,3,8,10,14","15,9,3,8,10,14","15,9,11,16,10,14","15,9,11,16,12,13","11,16,12,13,3,8"],
      'Validation Speakers':["15,9","11,16","12,13","3,8","10,14"],
      'Testing Speakers': ["11,16","12,13","3,8","10,14","15,9"]  
    })
    
    fold_infos = fold_infos.style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen','font-size': '12pt','text-align' : 'center',
                           'border-color': 'white'})
    st.write(fold_infos)

    st.markdown("<h3 style='text-align: center;font-family:courier new,garamond;'>Please select fold to view result:</h3>",unsafe_allow_html=True)
    
    fold = st.selectbox('', [0,1,2,3,4])

with col2:

    emotion_list =[]
    emotion_preds_list = []
    st.markdown("<h3 style='text-align: center;font-family:courier new;'>Test result for Fold {}</h3>".format(fold),unsafe_allow_html=True)
    checkpoint = torch.load(MODEL_PATH + 'fold' + str(fold))

    DEVICE  =torch.device("cpu")

    encoder = FeatureExtractor().to(DEVICE)
    emotion_classifier = EmotionClassifier().to(DEVICE)

    encoder_optimizer = torch.optim.Adam(encoder.parameters())
    emotion_optimizer = torch.optim.Adam(emotion_classifier.parameters())

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    emotion_classifier.load_state_dict(checkpoint['emotion_classifier_state_dict'])

    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    emotion_optimizer.load_state_dict(checkpoint['emotion_optimizer_state_dict'])

    emodb_dataset_test = EmoDBDataset2(ROOT,cv_index = fold, split= 'test')

    audio_df = emodb_dataset_test.df.copy()
    audio_df['File'] = audio_df['File'].str.replace('.pt','.wav')
    audio = audio_df['File'].values.tolist()

    TEST_BATCH_SIZE = 1 

    emodb_test_loader = DataLoader(dataset=emodb_dataset_test, batch_size= TEST_BATCH_SIZE, shuffle=False, drop_last=False,worker_init_fn=np.random.seed(42),num_workers=0)

    encoder.eval()
    emotion_classifier.eval()

    lemotion_testing_correct = 0 

    with st.spinner(text="Predicting..."): 

        with torch.no_grad():

            # 0 = anger, 1 = boredom, 2 = disgust, 3 = anxiety/fear, 4 = happiness, 5 = sadness, 6 = neutral
            emo_dict = {0 : 'anger', 1 : 'boredom' , 2 : 'disgust', 3 : 'anxiety', 4 : 'happiness', 
                        5 : 'sadness', 6 : 'neutral'}
            incorrect_index = []

            # 3. Testing Classification
            for index, (features, emotion, speaker) in enumerate(emodb_test_loader):
                features, emotion, speaker = features.to(DEVICE),emotion.to(DEVICE), speaker.to(DEVICE) 
                emotion_list.append(emo_dict[emotion.item()])
                conv_features = encoder(features)
                emotion_output = emotion_classifier(conv_features)
                _, emotion_preds = torch.max(emotion_output,1)
                emotion_preds_list.append(emo_dict[emotion_preds.item()])
                if emotion_preds!= emotion:
                    incorrect_index.append(index)
                lemotion_testing_correct += (emotion_preds == emotion).sum() 
                
            print(emotion_list)
            print(emotion_preds_list)
    # Accuracy
    accuracy = accuracy_score(emotion_list, emotion_preds_list)
    st.markdown("<h3 style='text-align: center;font-family:courier new;'> Emotion Predicted Correctly : {}/{} ({:.2f}%)</h3>".format(lemotion_testing_correct,(len(emodb_test_loader)*TEST_BATCH_SIZE),accuracy*100),unsafe_allow_html=True)        
    
    # Confusion Matrix
    st.markdown("<h3 style='text-align: center;font-family:courier new;'>Confusion Matrix</h3>",unsafe_allow_html=True)    
    confusion_matrix = confusion_matrix(emotion_list,emotion_preds_list,labels = ["anger", "boredom", "disgust","anxiety","happiness","sadness","neutral"])
    labels = ["anger", "boredom", "disgust","anxiety","happiness","sadness","neutral"]
    df_cm = pd.DataFrame(confusion_matrix, labels, labels)
    fig, ax = plt.subplots()

    fig = plt.figure(figsize=(8,6))
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0)

    
    ax = sns.heatmap(df_cm, annot=True, fmt="d",linewidths=.5,annot_kws={"size": 16})
    ax.tick_params(colors='w',grid_alpha=0)
    ax.set_yticklabels(labels,rotation=35,fontsize=16)
    ax.set_xticklabels(labels,rotation=35,fontsize=16)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.xaxis.label.set_color('cyan')
    ax.yaxis.label.set_color('cyan')
    ax.xaxis.label.set_fontsize('16')
    ax.yaxis.label.set_fontsize('16')

    st.write(fig)
     
    
with col3:    
    # Classification Report
    st.markdown("<h3 style='text-align: center;font-family:courier new;'>Detailed Statistics</h3>",unsafe_allow_html=True)    
    classification_report_df = pd.DataFrame(classification_report(emotion_list, emotion_preds_list,labels,output_dict=True)).T
    classification_report_df = classification_report_df.style.background_gradient(cmap='viridis')
    
    st.write(classification_report_df)
    
    # Grid
    st.markdown("<h3 style='text-align: center;font-family:courier new;'>Prediction result for each audio</h3>",unsafe_allow_html=True)    
    num_pad = 126 - len(emodb_test_loader)*TEST_BATCH_SIZE 
    neg_pad = np.ones(num_pad)*-0.5
    grids = np.zeros(len(emodb_test_loader)*TEST_BATCH_SIZE)
    grids = np.concatenate((grids,neg_pad))
    for index,element in enumerate(grids):
        if index in incorrect_index:
            grids[index] = 1
    
    grids = grids.reshape(3,-1)
    
    data = grids

    # create discrete colormap
    cmap = colors.ListedColormap(['grey','green', 'red'])
    bounds = [-1,0,0.5,1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig2, ax2 = plt.subplots()
    
    fig2.patch.set_facecolor('black')
    fig2.patch.set_alpha(0)
    ax2.imshow(data, cmap=cmap, norm=norm)

    ax2.grid(color='w')
    ax2.set_xticks(np.arange(-0.5, 42.5, 1))
    ax2.set_yticks(np.arange(-0.5, 3.5, 1))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    st.write(fig2)

    incorrect_df = pd.DataFrame({
      'Incorrect audio index': incorrect_index
    }).T
    
    incorrect_df = incorrect_df.style.set_properties(**{'background-color': 'black',
                           'color': 'red','font-size': '12pt','text-align' : 'center',
                           'border-color': 'white'})
    st.write(incorrect_df)

    with col4:
        st.markdown("<h3 style='text-align: center;font-family:courier new;'>Audio index for inspection:</h3>",unsafe_allow_html=True)
        #inspect_index = int(st.text_input(''))
        inspect_index = st.selectbox('', list(range(len(emodb_test_loader)*TEST_BATCH_SIZE)))

        audio_path = './Dataset/emodb/{}'.format(audio[inspect_index])
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/ogg')
        #st.markdown("<h3 style='text-align: center'>Actual emotion: {}</h3>".format(emotion_list[inspect_index]), unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;font-family:courier new;'>Actual:</h3>", unsafe_allow_html=True)
        st.image("{}.png".format(emotion_list[inspect_index]))
        # st.markdown("<h3 style='text-align: center'>Prediction emotion: {}</h3>".format(emotion_preds_list[inspect_index]), unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;font-family:courier new;'>Prediction:</h3>", unsafe_allow_html=True)
        st.image("{}.png".format(emotion_preds_list[inspect_index]))

st.balloons()