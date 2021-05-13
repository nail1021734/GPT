import torch
import math

class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + \
            torch.autograd.Variable(
                self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class GPT(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        dim_feedforward,
        dropout,
        layer_num,
        padding_idx,
        vocab_size
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )
        self.PE = PositionalEncoding(d_model, dropout)

        decoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.model = torch.nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=layer_num
        )

    def forward(self, batch_seq, src_mask):
        seq_len = batch_seq.shape[-1]
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1).to(batch_seq.device)
        x = self.embedding(batch_seq)
        x = x.transpose(0, 1)
        x = self.model(
            x,
            mask=mask,
            src_key_padding_mask=src_mask
        )
        x = x.transpose(0, 1)
        x = x @ self.embedding.weight.transpose(0, 1)
        x = torch.nn.functional.softmax(x, dim=-1)

        return x
