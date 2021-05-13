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

if __name__ == "__main__":
    # Set model config.
    cfg = Config(
        exp_name='tst',
        epoch=5,
        learning_rate=1e-2,
        batch_size=8,
        layer_num=3,
        d_model=256,
        dim_feedforward=1024,
        dropout=0.1,
        n_head=8,
        vocab_size=40000,
        seq_max_length=512,
        checkpoint_step=3000,
        log_step=10
    )
    cfg.save()

    # Set random seed.
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset and create data_loader
    dataset = LMNewsDataset(db_path='news.db')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=cfg.args.batch_size,
    )

    # Load tokenizer.
    tokenizer = BPETokenizer.load(cfg.args.exp_name)

    # Use bpe to Create tokenizer.
    # tokenizer = BPETokenizer()
    # tokenizer.train(
    #     dataset=dataset,
    #     vocab_size=cfg.args.vocab_size
    # )
    # tokenizer.save(cfg.args.exp_name)


    # Create model by config hyperparameter.
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

    # Set criterion and optimizer.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.args.learning_rate)


    # Init `SummaryWriter`.
    log_path = os.path.join('log', cfg.args.exp_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = torch.utils.tensorboard.SummaryWriter(log_path)

    iteration = 0
    for i in range(cfg.args.epoch):
        epoch_iterator = tqdm(
            data_loader,
            desc=f'epoch: {i}, loss: {0:.6f}'
        )
        total_loss = 0
        for x in epoch_iterator:
            iteration += 1

            # Tokenize input and get sequence of token_id and padding mask.
            seq_ids, _, batch_mask = tokenizer.batch_encode(
                sentence_list=x,
                max_length=cfg.args.seq_max_length
            )

            # Convert `List[int]` to `torch.Tensor`.
            src = torch.tensor([sample[:-1] for sample in seq_ids]).to(device)
            target = torch.tensor([sample[1:] for sample in seq_ids]).to(device)

            # Let mask into right form.
            batch_mask = torch.tensor(
                [sample[:-1] for sample in batch_mask],
                dtype=torch.bool
            ).logical_not().to(device)

            # Get model predict.
            pred = model(src, batch_mask)

            # Calculate loss.
            pred = pred.reshape(-1, tokenizer.tokenizer.get_vocab_size())
            target = target.reshape(-1)
            loss = criterion(pred, target)

            total_loss += loss.item()
            epoch_iterator.set_description(
                f'epoch: {i}, loss: {loss.item():.6f}'
            )

            # Update model parameter.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Output to tensorboard.
            if iteration %  cfg.args.log_step == 0:
                writer.add_scalar('loss', total_loss / cfg.args.log_step, iteration)
                total_loss = 0

            # Save checkpoint.
            if iteration % cfg.args.checkpoint_step == 0:
                save_path = os.path.join('checkpoint', cfg.args.exp_name)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_path, f'checkpoint-{iteration}.pt')
                )