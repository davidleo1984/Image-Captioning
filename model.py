import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        features = features.unsqueeze(1)
        caption_embed = self.embed(captions[:,:-1])
        inputs = torch.cat((features,caption_embed),1)
        out_lstm, states = self.lstm(inputs)
        out = self.fc(out_lstm)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        i=0
        caption=[]
        while i < max_len:
            out_lstm, states = self.lstm(inputs, states)
            out = self.fc(out_lstm)
            idx = out.max(2)[1]
            caption.append(idx.item())
            if idx.item() == 1:
                break
            inputs = self.embed(idx)
            i += 1
        
        return caption
        
        
        
        
        