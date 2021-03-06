import os
import json
import argparse


class Config:
    def __init__(
        self,
        exp_name,
        epoch,
        learning_rate,
        batch_size,
        layer_num,
        d_model,
        dim_feedforward,
        dropout,
        n_head,
        vocab_size,
        seq_max_length,
        checkpoint_step,
        log_step,
        update_step
    ):
        self.cfg_dict = {
            'exp_name': exp_name,
            'epoch': epoch,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'layer_num': layer_num,
            'd_model': d_model,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'n_head': n_head,
            'vocab_size': vocab_size,
            'seq_max_length': seq_max_length,
            'checkpoint_step': checkpoint_step,
            'log_step': log_step,
            'update_step': update_step
        }

        self.args = argparse.Namespace(**self.cfg_dict)

    def save(self):
        cfg_path = os.path.join('config', f'{self.args.exp_name}.json')
        json.dump(self.cfg_dict, open(cfg_path, 'w', encoding='utf8'))

    @staticmethod
    def load(exp_name):
        cfg_path = os.path.join('config', f'{exp_name}.json')
        cfg_dict = json.load(open(cfg_path, 'r', encoding='utf8'))
        return Config(**cfg_dict)
