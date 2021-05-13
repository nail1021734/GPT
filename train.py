import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from config import Config
from dataset import LMNewsDataset
from model import GPT
from tokenizer import BPETokenizer

if __name__ == "__main__":
    cfg = Config(
        exp_name='tst',
        epoch=5,
        learning_rate=1e-4,
        batch_size=32,
        layer_num=1,
        d_model=256,
        dim_feedforward=1024,
        dropout=0.1,
        n_head=8,
        vocab_size=10000,
        seq_max_length=512
    )
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = LMNewsDataset(db_path='news.db')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.args.batch_size,
    )

    tokenizer = BPETokenizer.load(cfg.args.exp_name)

    model = GPT(
        d_model=cfg.args.d_model,
        n_head=cfg.args.n_head,
        dim_feedforward=cfg.args.dim_feedforward,
        dropout=0.1,
        layer_num=cfg.args.layer_num,
        padding_idx=tokenizer.tokenizer.token_to_id('[PAD]'),
        vocab_size=tokenizer.tokenizer.get_vocab_size()
    )

    model = model.to(device)
    model.train()
    model.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.args.learning_rate)

    for i in range(cfg.args.epoch):
        epoch_iterator = tqdm(
            data_loader,
            desc=f'epoch: {i}, loss: {0:.6f}'
        )
        for x in epoch_iterator:
            seq_ids, seq_tokens, batch_mask = tokenizer.batch_encode(
                sentence_list=x,
                max_length=cfg.args.seq_max_length
            )
            src = torch.tensor([sample[:-1] for sample in seq_ids]).to(device)
            target = torch.tensor([sample[1:] for sample in seq_ids]).to(device)
            batch_mask = torch.tensor(
                [sample[:-1] for sample in batch_mask],
                dtype=torch.bool
            ).to(device)
            pred = model(src, batch_mask)
            pred = pred.reshape(-1, tokenizer.tokenizer.get_vocab_size())
            target = target.reshape(-1)
            loss = criterion(pred, target)

            epoch_iterator.set_description(
                f'epoch: {i}, loss: {loss.item():.6f}'
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
