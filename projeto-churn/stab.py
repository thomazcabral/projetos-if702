import torch
import torch.nn as nn
import torch.nn.functional as F

class DropConnectLinear(nn.Module):
    def __init__(self, in_features, out_features, dropconnect_rate=0.5, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropconnect_rate = dropconnect_rate

    def forward(self, x):
        # O DropConnect aplica o dropout diretamente na matriz de pesos durante o treino
        if self.training and self.dropconnect_rate > 0.0:
            weight = F.dropout(self.linear.weight, p=self.dropconnect_rate, training=self.training)
        else:
            weight = self.linear.weight
        return F.linear(x, weight, self.linear.bias)

class STabClassifier(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, embed_dim, dropout_rate=0.1, dropconnect_rate=0.5, learning_rate=0.001, batch_size=32, random_state=42, **kwargs):
        super().__init__()

        if random_state is not None:
            torch.manual_seed(random_state)
            
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        transformer_out_dim = input_dim * embed_dim
        
        self.fc = nn.Sequential(
            DropConnectLinear(transformer_out_dim, 512, dropconnect_rate=dropconnect_rate),
            nn.ReLU(),
            
            DropConnectLinear(512, 128, dropconnect_rate=dropconnect_rate),
            nn.ReLU(),
            
            DropConnectLinear(128, 1, dropconnect_rate=dropconnect_rate)
        )

    def forward(self, x):
        embedded_features = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.feature_embeddings)]
        
        x_emb = torch.stack(embedded_features, dim=1)
        
        attn_out = self.transformer(x_emb)
        
        flat_out = attn_out.view(attn_out.size(0), -1)
        
        return self.fc(flat_out)