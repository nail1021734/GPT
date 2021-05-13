import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, dataset, vocab_size):
        # Get BPE trainer.
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        # Train tokenizer.
        self.tokenizer.train_from_iterator(dataset, trainer)

    def encode(self, sentence):
        r"""
        Use `output.id` to get sequence of token_id.
        Use `output.tokens` to get sequence of token.
        """

        output = self.tokenizer.encode(sentence)
        return output

    def batch_encode(self, sentence_list, max_length):
        r"""
        Return 3 items.
        First item is batch of token_id sequence.
        Second item is batch of token sequence.
        Third item is padding_mask of batch.
        """
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=max_length)
        output = self.tokenizer.encode_batch(sentence_list)
        batch_token_ids = [seq.ids for seq in output]
        batch_tokens = [seq.tokens for seq in output]
        batch_mask = [seq.attention_mask for seq in output]

        return batch_token_ids, batch_tokens, batch_mask

    def decode(self, token_ids):
        r"""
        `token_ids` is a list of int.
        Return a `str`.
        """
        return self.tokenizer.decode(
            ids=token_ids,
            skip_special_tokens=True
        )

    def batch_decode(self, token_ids_list):
        r"""
        Return a list of `str`.
        """
        return self.tokenizer.decode_batch(
            sequences=token_ids_list,
            skip_special_tokens=True
        )

    def save(self, filename):
        save_path = os.path.join('tokenizer', f'{filename}.json')
        self.tokenizer.save(save_path)

    @staticmethod
    def load(filename):
        load_path = os.path.join('tokenizer', f'{filename}.json')
        tokenizer = BPETokenizer()
        tokenizer.tokenizer = Tokenizer.from_file(load_path)
        return tokenizer
