# FYP_Bryan-Leow-Xuan-Zhen
This Repository contains the implementation of my Final Year Project "Speaker Invariant Speech Emotion Recognition with Domain Adversarial Training".

The project aims to create a model architecture for Speaker Invariant Emotion Recognition (SIER) task by using Domain Adversarial Training (DAT) as a framework. 

2 different datasets are used for the experiments, EmoDB and Ravdess.

Comparisons made in these experiments include:

1. Our 2D-CNN-biGRU encoder with DAT vs without DAT
2. Our 2D-CNN-biGRU encoder vs TDNN-BiLSTM encoder introduced in Tu et. al
3. Our 2D-CNN-biGRU encoder vs 1D CNN GRU Encoder in Li et. al
4. Mel-Frequency Cepstral Coefficient (MFCC) features vs Log Mel Spectrogram (LMS) features for SIER tasks and DAT with our encoder

Results show that our 2D-CNN-biGRU encoder has consistently outperform the other SIER encoders using DAT and that MFCC features work betters than LMS features with our encoder in this project. It also shows that the depth of the encoder,the degree and rate of domain adapatation will have significant effect on the performance on the model architecture.
