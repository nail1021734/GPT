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
    input = torch.tensor(ids).unsqueeze(dim=0).to(device)
    for _ in range(max_length):
        output = model.predict(input)
        new_id = output.argmax(dim=-1)[:, -1]
        input = torch.cat((input, new_id.unsqueeze(dim=0)), dim=-1)
    return tokenizer.decode(input.tolist()[0])

if __name__ == '__main__':
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    cfg = Config.load('exp14_adamW_v2.3')

    tokenizer = BPETokenizer.load(cfg.args.exp_name)

    model = load_model(cfg.args.exp_name, 'checkpoint-415000.pt')
    model = model.to(device)
    model.eval()

    result = top1_inference(
        prefix='[SEP]<num>歲童柔道被摔腦死!父淚聽醫曝最壞結果「恐成植物人」[SEP]台中小一<num>歲男童,因上柔道課被教練指派由「學長」過肩摔',
        tokenizer=tokenizer,
        device=device,
        model=model,
        max_length=100
    )
    print(result)