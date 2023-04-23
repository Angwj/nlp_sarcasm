import torch
import torchvision.transforms as transforms
import opensmile
import numpy as np
import librosa
import speech_recognition as SR
# from Katna.video import Video
# from Katna.writer import KeyFrameDiskWriter
from moviepy.editor import VideoFileClip
import tempfile
import ffmpeg
import subprocess
from io import BytesIO
from PIL import Image
import cv2
import io
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor():
    def __init__(self,tokenizer, model, resnet152):
        self.name = "FeatureExtractor"
        self.recognizer = SR.Recognizer()
        #Load AutoModel from huggingface model repository
        self.tokenizer = tokenizer
        self.model = model

        self.smile = opensmile.Smile(
                        feature_set=opensmile.FeatureSet.GeMAPSv01b,
                        feature_level=opensmile.FeatureLevel.Functionals,
                    )
        
        # Load the pretrained model
        self.resnet152 = resnet152

        # Image transforms
        self.transf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    #-----------------------------------------------#
    #                  TEXT
    #-----------------------------------------------#
    # Function to extract features from a given text
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def extractTextFeatures(self,text):
        #Tokenize sentences
        encoded_input = self.tokenizer([text],
                                        add_special_tokens = True, 
                                        padding='max_length',
                                        max_length = 100,
                                        truncation=True,
                                        return_tensors='pt')

        #Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        return sentence_embeddings.squeeze()