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
    
        # self.video = Video()

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
        # return model_output
    
    #-----------------------------------------------#
    #                  AUDIO
    #-----------------------------------------------#
    # Function to extract features from a given audio
    def transcript(self,audio_buffer):
        try:
            audio_buffer.seek(0)
            aud = SR.AudioData(audio_buffer.read(), sample_rate=44100, sample_width=2)
            text = self.recognizer.recognize_google(aud)
        except:
            text = ""
        return text

    def extractAudioFeatures(self, audio_buffer):

        # Load the audio file with sampling rate of 22.5kHz
        y, sr = librosa.load(audio_buffer,sr=22500)
  
        # Remove vocals first
        D = librosa.stft(y, hop_length=512)
        S_full, phase = librosa.magphase(D)

        S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric="cosine")

        S_filter = np.minimum(S_full, S_filter)

        margin_i, margin_v = 2, 4
        power = 2
        mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
        S_foreground = mask_v * S_full

        # Recreate vocal_removal y
        new_D = S_foreground * phase
        y = librosa.istft(new_D)

        n_fft = int(sr * 0.1)
        hop_length = n_fft              # non-overlapping, hop_length = n_fft//2 if overlapping

        # print(n_fft,hop_length)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        mels = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        spcen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)

        #calculate the first-order differences of adjacent frames along the specified
        mfccs_delta = librosa.feature.delta(mfccs,mode='nearest')
        mels_delta = librosa.feature.delta(mels,mode='nearest')
        spcen_delta = librosa.feature.delta(spcen,mode='nearest')
        # print(mfccs_delta.shape,mels_delta.shape,spcen_delta.shape)

        prosodic = []
        
        frames = librosa.util.frame(y, frame_length=hop_length, hop_length=hop_length, axis=0)
        for frame in frames:
            prosody = self.smile.process_signal(frame,sr)
            pros = np.array(prosody)
            prosodic.append(pros.flatten())
        prosodic = np.array(prosodic)
        prosodic = prosodic.T
        # print(prosodic.shape)

        feature_arrays = [mfccs, mfccs_delta, mels, mels_delta, spcen, spcen_delta]
        for i, feature in enumerate(feature_arrays):
            feature_arrays[i] = feature[:,:prosodic.shape[1]]
        mfccs, mfccs_delta, mels, mels_delta, spcen, spcen_delta = feature_arrays
        # print(mfccs.shape,mels.shape,spcen.shape)

        feature = np.concatenate([mfccs, mels, spcen, mfccs_delta, mels_delta, spcen_delta,prosodic], axis=0)
        feature = np.mean(feature, axis = 1)

        text = self.transcript(audio_buffer)
        # text_feature = self.extractTextFeatures(text)

        return feature.transpose(), text

    #-----------------------------------------------#
    #                  VIDEO
    #-----------------------------------------------#

    def get_frames(self,frames_folders_path):
        # Get all frame file names
        frames = None
        frames_folder = os.listdir(frames_folders_path)

        #each frames
        for i,frame_file_name in enumerate(frames_folder):
            frame = Image.open(os.path.join(frames_folders_path, frame_file_name))
            #Transform image to standard size
            frame = self.transf(frame)

            if frames is None:
                frames = torch.empty((len(frames_folder), *frame.size()))
                frames[i] = frame
        return frames

    def frames_features(self,frames_folder_path):
        frames = self.get_frames(frames_folder_path)
        frames = frames.to(DEVICE)
        # Run the model on input data
        output = []
        batch_size = 10                
        for start_index in range(0, len(frames), batch_size):
            end_index = min(start_index + batch_size, len(frames))
            frame_range = range(start_index, end_index)
            frame_batch = frames[frame_range]
            avg_pool_value = self.resnet152(frame_batch)
            output.append(avg_pool_value.detach().cpu().numpy())

        output = np.concatenate(output)

        return output

    # Function to extract features from a given video
    def extractVideoFeatures(self, video_path):
        print("Extracting video features from", video_path)
        # Save key frames to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpframes = os.path.join(tmpdir, "frames")
            print("Saving key frames to", tmpframes)
            print(os.path.exists(tmpframes))
            os.makedirs(tmpframes)
            print(os.path.exists(tmpframes))

            # # Extract key frames
            # diskwriter = KeyFrameDiskWriter(location=tmpframes)
            # self.video.extract_video_keyframes(
            #     no_of_frames=10, file_path=video_path, writer=diskwriter
            # )

            # # If no keyframes were extracted, extract them manually
            # if len(os.listdir(tmpframes)) == 0:
            #     # Open the video file
            cap = cv2.VideoCapture(video_path)
            count = 0
            copy = 0
            # Loop through the video frames
            while cap.isOpened():
                # Read the next frame
                ret, frame = cap.read()

                if not ret:
                    break
                # Save every 15th frame as a keyframe
                if (count % 15 == 0) and (copy <=10 ):
                    cv2.imwrite(tmpframes+f'/{os.path.basename(tmpframes)}_{copy}.jpg', frame)
                    copy +=1
                count += 1
            # Release the video file
            cap.release()

            features = self.frames_features(tmpframes)
            vid_features = np.mean(features, axis = 0)
            print("Extracted feat")
            print(os.path.exists(tmpframes))
            
            audio_path = os.path.join(tmpframes, "audio.wav")

            # subprocess.call(['ffmpeg', '-i', video_path, '-codec:a', 'pcm_s16le','-ac', '1', audio_path])
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path)
            print(os.path.exists(audio_path))
            video.close()

            with open(audio_path, 'rb') as file:
                # Create a ByteIO object and write the contents of the file to it
                audio_buffer = io.BytesIO(file.read())

            audio_features, text = self.extractAudioFeatures(audio_buffer)

        return vid_features, audio_features, text