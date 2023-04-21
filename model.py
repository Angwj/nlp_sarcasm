import torch.nn.functional as F
import torch.nn as nn
import torch

T_DIM, V_DIM, A_DIM = 768, 2048, 360

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))
        return y_3
    
# h+i, +c
class LF_DNN1(nn.Module):
    def __init__(self, sn_dropout=0.3, fusion_dropout=0.4):
        super(LF_DNN1, self).__init__()
        # Define dim size of SubNet
        self.text_in, self.video_in, self.audio_in = T_DIM, V_DIM, A_DIM
        self.text_context_in, self.video_context_in, self.audio_context_in = T_DIM, V_DIM, A_DIM

        self.text_hidden, self.video_hidden, self.audio_hidden = 128, 128, 32
        self.text_context_hidden, self.video_context_hidden, self.audio_context_hidden = 32, 32, 16

        self.post_fusion_dim1 = 256
        self.post_fusion_dim2 = 32

        self.video_prob, self.text_prob, self.audio_prob, self.post_fusion_prob = (sn_dropout, sn_dropout, sn_dropout, fusion_dropout)
        self.video_context_prob, self.text_context_prob, self.audio_context_prob = (sn_dropout, sn_dropout, sn_dropout)

        # SubNet 
        # Utterance
        self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_prob)
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob) 
        # Context
        self.video_context_subnet = SubNet(self.video_context_in, self.video_context_hidden, self.video_context_prob)
        self.audio_context_subnet = SubNet(self.audio_context_in, self.audio_context_hidden, self.audio_context_prob)
        self.text_context_subnet = SubNet(self.text_context_in, self.text_context_hidden, self.text_context_prob)

        # Late Fusion
        self.post_fusion_dropout1 = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_dropout2 = nn.Dropout(p=self.post_fusion_prob)

        self.post_fusion_layer_1 = nn.Linear( self.text_in + self.audio_in + 
                                             self.video_in + self.text_hidden + self.audio_hidden +
                                             self.video_hidden, self.post_fusion_dim1)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim1, self.post_fusion_dim1)

        self.post_fusion_layer_4 = nn.Linear(self.post_fusion_dim1 + self.text_context_hidden + self.audio_context_hidden +
                                             self.video_context_hidden, self.post_fusion_dim2)
        
        self.post_fusion_layer_5 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)
        self.post_fusion_layer_6 = nn.Linear(self.post_fusion_dim2, self.post_fusion_dim2)
        self.post_fusion_layer_7 = nn.Linear(self.post_fusion_dim2, 2)

    def forward(self, text_x, audio_x, video_x, text_context_x, audio_context_x, video_context_x):
        text_h = self.text_subnet(text_x)
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_c_h = self.text_context_subnet(text_context_x)
        audio_c_h = self.audio_context_subnet(audio_context_x)
        video_c_h = self.video_context_subnet(video_context_x)
        
        # 
        fusion_h = torch.cat([text_x, audio_x, video_x, text_h, audio_h, video_h], dim=-1) 
        x = self.post_fusion_dropout1(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True) 
        x = F.relu(self.post_fusion_layer_2(x), inplace=True) 
        x = F.relu(self.post_fusion_layer_3(x), inplace=True)

        x = torch.cat([x, text_c_h, audio_c_h, video_c_h], dim=-1)
        # x = torch.cat([x, text_c_h], dim=-1)
        x = self.post_fusion_dropout2(x)
        x = F.relu(self.post_fusion_layer_4(x), inplace=True)
        x = F.relu(self.post_fusion_layer_5(x), inplace=True)
        x = F.relu(self.post_fusion_layer_6(x), inplace=True)
    
        output = self.post_fusion_layer_7(x)
        return output