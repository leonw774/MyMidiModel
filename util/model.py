import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .corpus import FEATURE_INDEX, INPUT_FEATURE_NAME, OUTPUT_FEATURE_NAME, array_to_text_list, text_list_to_array
from .midi import piece_to_midi
from .tokens import PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR

class MidiTransformerDecoder(nn.Module):
    def __init__(self,
            vocabs: dict,
            max_seq_length: int,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            ) -> None:
        super().__init__()

        self._pad_id = vocabs['events']['text2id'][PADDING_TOKEN_STR]
        # self._pad_id = 0
        assert self._pad_id == 0

        self._embedding_dim = embedding_dim
        self._max_seq_length = max_seq_length
        self._vocabs = vocabs
        self._end_token_id = vocabs['events']['text2id'][END_TOKEN_STR]

        # Input features
        self.embedding_vocabs_size = [
            ( vocabs[feature_name]['size'] if feature_name != 'measure_numbers' else max_seq_length )
            for feature_name in INPUT_FEATURE_NAME
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                padding_idx=self._pad_id # [...] the embedding vector at padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        # Output features
        self.logit_vocabs_size = [
            vocabs[feature_name]['size'] for feature_name in OUTPUT_FEATURE_NAME
        ]
        if vocabs['paras']['position_method'] == 'event':
            self.logit_vocabs_size.pop()
        self.output_features_indices = [
            FEATURE_INDEX[feature_name]
            for feature_name in OUTPUT_FEATURE_NAME
        ]
        self.logits = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dim,
                out_features=vsize
            )
            for vsize in self.logit_vocabs_size
        ])

        layer = nn.TransformerEncoderLayer( # name's encoder, used as decoder. Cause we don't need memory
            d_model=embedding_dim,
            nhead=attn_heads_number,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=layers_number
        )

    def to_output_features(self, input_seq_features):
        # input_seq_features.shape = (batch, sequence, in_feature_num)
        # return (batch, sequence, out_feature_num)
        return input_seq_features[..., self.output_features_indices]

    def forward(self, x, mask):
        # x.shape = (batch_size, seq_size, feature)
        try:
            x = sum(
                emb(x[:,:,i]) for i, emb in enumerate(self.embeddings)
            )
        except:
            for i, emb_size in enumerate(self.embedding_vocabs_size):
                print(torch.all(x[:,:,i] < emb_size))
            exit(1)
        x = self.transformer_decoder(
            src=x,
            mask=mask
        )
        logits = [
            logit(x) for logit in self.logits
        ]
        return logits

    @torch.no_grad()
    def generate(self, start_seq, steps, temperature=1.0) -> list:
        """
            return the text list of generated piece
        """
        training_state = self.training

        if len(start_seq.shape) != 3:
            raise ValueError(f'start_seq\'s shape have to be (1, seq_length, input_feature_num), get {start_seq.shape}')
        elif start_seq.shape[0] != 1 or start_seq.shape[2] != len(self.input_features_name):
            raise ValueError(f'start_seq\'s shape have to be (1, seq_length, input_feature_num), get {start_seq.shape}')

        max_length_mask = get_seq_mask(self._max_seq_length).to(start_seq.device)

        input_seq = start_seq
        output_seq = self.to_output_features(start_seq)
        output_text_list = []
        end_with_end_token = False
        # print(output_seq.shape)
        for _ in tqdm(range(steps)):
            clipped_input_seq = input_seq[:, -self._max_seq_length:]
            mask = max_length_mask[:clipped_input_seq.shape[1], :clipped_input_seq.shape[1]]
            logits = self.forward(clipped_input_seq, mask)
            last_logits = [
                l[:, -1, :] # l has shape (1, sequence, feature_vocab_size)
                for l in logits
            ]
            try_count = 0
            try_count_limit = 1000
            while try_count < try_count_limit:
                sampled_features = [
                    torch.multinomial(F.softmax(l[0] / temperature, dim=0), 1) 
                    for l in last_logits # l has shape (1, feature_vocab_size)
                ]
                new_token = torch.stack(sampled_features, dim=1) # shape = (1, feature_num)
                new_token = torch.unsqueeze(new_token, dim=0) # shape = (1, 1, feature_num)
                new_output_seq = torch.cat((output_seq, new_token), dim=1)
                new_output_text_list = array_to_text_list(new_output_seq[0].cpu().numpy(), vocabs=self._vocabs, is_input=False)

                try:
                    # for format checking
                    piece_to_midi(' '.join(new_output_text_list), self._vocabs['paras']['nth'])
                    # for format checking and position, tempo, measure_number calculation
                    input_seq = torch.from_numpy(
                        text_list_to_array(new_output_text_list, vocabs=self._vocabs)
                    ).unsqueeze(0).long()
                except:
                    try_count += 1
                    continue # keep sampling until no error

                output_seq = new_output_seq
                output_text_list = new_output_text_list
                break

            if try_count == try_count_limit:
                break

            if sampled_features[FEATURE_INDEX['evt']] == self._end_token_id:
                end_with_end_token = True
                break

        if not end_with_end_token:
            output_text_list.append(END_TOKEN_STR)

        self.train(training_state)
        return output_text_list


def calc_losses(pred_logit, target):
    """
        pred is a list
        - length=out_feature_num
        - elements are tenors with shape=(batch_size, seq_size, feature_vocab_size)
        target has shape: (batch_size, seq_size, out_feature_num)
    """
    # basically treat seq_size as the K in
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
    # target have to be long
    target = target.long()
    head_losses = [
        F.cross_entropy(
            input=pred_feature_logit.transpose(1, 2), # (batch_size, feature_vocab_size, seq_size)
            target=target[..., i] # (batch_size, seq_size)
        )
        for i, pred_feature_logit in enumerate(pred_logit)
    ]
    return head_losses
    # loss = sum(head_losses)
    # return loss

def calc_permutable_subseq_losses(pred_logit, target, mps_indices):
    """
        pred is a list
        - length: out_feature_num
        - elements are tenors with shape: (batch_size, seq_size, feature_vocabs_size)
        target is tensor with shape: (batch_size, seq_size, out_feature_num)
        mps_indices is numpy object array of numpy int16 arrays in varying lengths
    """
    raise NotImplementedError('calc_permutable_subseq_loss not implemented yet.')


def get_seq_mask(size: int):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()
