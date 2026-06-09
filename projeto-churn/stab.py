import torch
import torch.nn as nn

class STabClassifier(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, embed_dim, dropout_rate, learning_rate=0.001, batch_size=32, random_state=42):
        super(STabClassifier, self).__init__()
        if random_state is not None:
            torch.manual_seed(random_state)
            
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(input_dim)
        ])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        embedded_features = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.feature_embeddings)]
        x_emb = torch.cat([f.unsqueeze(1) for f in embedded_features], dim=1)
        attn_out = self.transformer(x_emb)
        flat_out = attn_out.reshape(attn_out.size(0), -1)
        return self.fc(flat_out)