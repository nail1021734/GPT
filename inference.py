import argparse
import torch
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from dataset import LMNewsDataset
from model import GPT
from tokenizer import BPETokenizer


def load_model(exp_name, filename):
    ckp_path = os.path.join('checkpoint', exp_name, filename)

    model = GPT(
        d_model=cfg.args.d_model,
        n_head=cfg.args.n_head,
        dim_feedforward=cfg.args.dim_feedforward,
        dropout=cfg.args.dropout,
        layer_num=cfg.args.layer_num,
        padding_idx=tokenizer.tokenizer.token_to_id('[PAD]'),
        vocab_size=tokenizer.tokenizer.get_vocab_size()
    )
    model.load_state_dict(torch.load(ckp_path, map_location='cpu'))
    return model

def top1_inference(prefix, tokenizer, device, model ,max_length):
    ids = tokenizer.encode(prefix).ids
    inputs = torch.tensor(ids).unsqueeze(dim=0).to(device)
    for _ in range(max_length):
        output = model.predict(inputs)
        new_id = output.argmax(dim=-1)[:, -1]
        inputs = torch.cat((inputs, new_id.unsqueeze(dim=0)), dim=-1)
    return tokenizer.decode(inputs.tolist()[0])

def topk_inference(prefix, tokenizer, device, model ,max_length, topk):
    ids = tokenizer.encode(prefix).ids
    inputs = torch.tensor(ids).unsqueeze(dim=0).to(device)
    for _ in range(max_length):
        output = model.predict(inputs)[:, -1, :]
        topk_prob = output.topk(k=topk, dim=-1)
        select_id = torch.multinomial(topk_prob.values, 1).item()
        new_id = topk_prob.indices[..., select_id]
        inputs = torch.cat((inputs, new_id.unsqueeze(dim=0)), dim=-1)
    return tokenizer.decode(inputs.tolist()[0])

if __name__ == '__main__':
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    cfg = Config.load('ettoday')

    tokenizer = BPETokenizer.load(cfg.args.exp_name)

    model = load_model(cfg.args.exp_name, 'checkpoint-670000.pt')
    model = model.to(device)
    model.eval()

    result = topk_inference(
        prefix='[TITLE]永遠覺得自己不夠好.. 阻礙成功的「<num>種性格」你中了?[ARTICLE]',
        tokenizer=tokenizer,
        device=device,
        model=model,
        max_length=100,
        topk=5,
    )
    print(result)