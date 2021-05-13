import torch


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
        key_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        key_mask = torch.triu(key_mask, diagonal=1).to(batch_seq.device)
        x = self.embedding(batch_seq)
        print(x)
        x = x.transpose(0, 1)
        x = self.model(
            x,
            mask=key_mask,
            src_key_padding_mask=src_mask
        )
        x = x.transpose(0, 1)
        x = x @ self.embedding.weight.transpose(0, 1)
        x = torch.nn.functional.softmax(x, dim=-1)

        return x
