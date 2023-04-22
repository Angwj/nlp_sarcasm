import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from featureExtractor import FeatureExtractor
from transformers import AutoTokenizer, AutoModel
import tempfile
from model import LF_DNN1
from io import BytesIO
import os

T_DIM = 768
A_DIM = 360
V_DIM = 2048
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[-2, -1])

#@st.cache() decorator to avoid reloading the model each time 
@st.cache
def load_extractor():
    pretrained_model = 'bert-base-uncased'
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModel.from_pretrained(pretrained_model)

    resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    resnet152 = nn.Sequential(*list(resnet152.children())[:-2], GlobalAvgPool())
    resnet152.eval()
    resnet152 = resnet152.to(DEVICE)
    extractor = FeatureExtractor(tokenizer=tokenizer, model=model, resnet152=resnet152)
    return extractor

def predict(text, audio, visual, text_c, audio_c, visual_c):
        # Load the trained multimodal model
        model_path = './LateFusion/lf_models/lf_adam.Adam.pth'
        model = LF_DNN1(sn_dropout=0.2, fusion_dropout=0.3)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            text = torch.unsqueeze(text.clone().detach().to(DEVICE), 0).float()
            audio = torch.unsqueeze(audio.clone().detach().to(DEVICE), 0).float()
            visual = torch.unsqueeze(visual.clone().detach().to(DEVICE), 0).float()
            text_c = torch.unsqueeze(text_c.clone().detach().to(DEVICE), 0).float()
            audio_c = torch.unsqueeze(audio_c.clone().detach().to(DEVICE), 0).float()
            visual_c = torch.unsqueeze(visual_c.clone().detach().to(DEVICE), 0).float()
            prediction = model(text, audio, visual, text_c, audio_c, visual_c)
            prob = torch.softmax(prediction, dim=1)
            prob = prob[0][1].item()
            
        return torch.argmax(prediction).item() , round(prob, 2)
    

extractor = load_extractor()
print('loaded')



st.set_page_config(
    page_title="Sarcasm Detector",
    page_icon=":smirk:",
    layout="wide",
)

# Sidebar
st.sidebar.title("About")
st.sidebar.write("---")
st.sidebar.write("""The sarcasm detector uses text, audio and visual features to predict whether a given utterance is sarcastic or not.
                You can provide short video clips from sitcoms or any TV shows like The Big Bang Theory, Friends, etc.""")
st.sidebar.image("https://i.gifer.com/8SUl.gif", width=300, use_column_width=None)

# Get user input
st.title("Multimodal Sarcasm Detection")
"---"
input_type = st.selectbox("Input Type", ["Text", "Audio", "Video"])
context = st.checkbox("Do you have more context?")
"---"

col1, col2 = st.columns(2)
#----------------------------------------------#
#              Utterance input                 #
#----------------------------------------------#
col1.subheader("Utterance")
if input_type == "Text":
    input_data = col1.text_area("Enter text (max 100 words):", max_chars=100,key='utterance')

elif input_type == "Audio":
    input_data = col1.file_uploader("Upload audio file (mp3, wav):", type=["mp3", "wav"],key='utterance')
    if input_data is not None:
        # Save audio as buffer to allow multiple references
        audio_buffer_u = BytesIO(input_data.read())
        col1.audio(audio_buffer_u.getvalue(), format='audio/' + input_data.type.split('/')[1])
        audio_buffer_u.seek(0)
        A_u, T_u = extractor.extractAudioFeatures(audio_buffer_u)
        col1.info("Text transcripted: "+T_u)

else:
    input_data = col1.file_uploader("Upload video file (mp4, avi):", type=["mp4", "avi", "mov"],key='utterance')
    if input_data is not None:
        suffix = input_data.type
        suffix = suffix.split('/')[1]
        # Save video file to temporary directory
        tmpvideo = './tmp/tmpvideo.'+suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+suffix) as tmpvideo:
            tmpvideo.write(input_data.read())
            video_file_u = tmpvideo.name
        col1.video(video_file_u)
        V_u, A_u, T_u = extractor.extractVideoFeatures(video_file_u)

#----------------------------------------------#
#               Context input                  #
#----------------------------------------------#
input_data2 = None
if context == True:
    col2.subheader("Context")
    if input_type == "Text":
        input_data2 = col2.text_area("Enter text (max 100 words):", max_chars=100,key='context')

    elif input_type == "Audio":
        input_data2 = col2.file_uploader("Upload audio file (mp3, wav):", type=["mp3", "wav"],key='context')
        if input_data2 is not None:
            # Save audio as buffer to allow multiple references
            audio_buffer_c = BytesIO(input_data2.read())
            col2.audio(audio_buffer_c.getvalue(), format='audio/' + input_data2.type.split('/')[1])
            audio_buffer_c.seek(0)
            A_c, T_c = extractor.extractAudioFeatures(audio_buffer_c)
            col2.info("Text transcripted: "+T_c)

    else:
        input_data2 = col2.file_uploader("Upload video file (mp4, avi):", type=["mp4", "avi", "mov"],key='context')
        if input_data2 is not None:
            # Save file to temporary directory
            suffix = input_data2.type
            suffix = suffix.split('/')[1]

            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+suffix) as tmpvideo2:
                tmpvideo2.write(input_data2.read())
                video_file_c = tmpvideo2.name
            col2.video(video_file_c)

# Predict button
if st.button("Predict"):
    if input_data:
        if input_type == "Text":
            T_u = extractor.extractTextFeatures(input_data)
            # Pad zeros to match the dimensions of the model
            A_u = torch.zeros(A_DIM)
            V_u = torch.zeros(V_DIM)

            if input_data2:
                T_c = extractor.extractTextFeatures(input_data2)
            else:
                T_c = torch.zeros(T_DIM)
            A_c = torch.zeros(A_DIM)
            V_c = torch.zeros(V_DIM)

        elif input_type == "Audio":
            T_u = extractor.extractTextFeatures(T_u)
            A_u = torch.tensor(A_u)
            V_u = torch.zeros(V_DIM)
            
            if input_data2:
                T_c = extractor.extractTextFeatures(T_c)  
                A_c = torch.tensor(A_c) 
                
            else:
                T_c = torch.zeros(T_DIM)
                A_c = torch.zeros(A_DIM)
            V_c = torch.zeros(V_DIM)

        else:
            # V_u, A_u, T_u = extractor.extractVideoFeatures(video_file_u)
            T_u = extractor.extractTextFeatures(T_u)
            A_u = torch.tensor(A_u)
            V_u = torch.tensor(V_u)

            if input_data2:
                V_c, A_c, T_c = extractor.extractVideoFeatures(video_file_c)
                T_c = extractor.extractTextFeatures(T_c)
                A_c = torch.tensor(A_c)
                V_c = torch.tensor(V_c)
                os.remove(video_file_c)
            else:
                T_c = torch.zeros(T_DIM)
                A_c = torch.zeros(A_DIM)
                V_c = torch.zeros(V_DIM)
            # remove temporary video file
            os.remove(video_file_u)
            
        # print(T_u.dtype, A_u.dtype, V_u.dtype, T_c.dtype, A_c.dtype, V_c.dtype)
        # print(T_u.shape, A_u.shape, V_u.shape, T_c.shape, A_c.shape, V_c.shape)
        is_sarcastic, prob = predict(T_u, A_u, V_u, T_c, A_c, V_c)

        if is_sarcastic:
            st.success("SARCASM DETECTED!!!"+ " with probability: "+str(prob))
        else:
            st.info("No sarcasm detected.")
    else:
        st.warning("Please provide input for the selected modality.")