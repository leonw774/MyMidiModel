from torch import nn, triu, ones
from torch.nn.functional import cross_entropy

from util import SPECIAL_TOKENS

class MidiTransformerDecoder(nn.Module):
    def __init__(self,
            layers_number: int,
            attn_heads_number: int,
            d_model: int,
            vocabs: dict,
            ) -> None:
        super().__init__()

        self._pad_id = vocabs['events']['text2id'][SPECIAL_TOKENS[0]]
        # self._pad_id = 0
        assert self._pad_id == 0

        self._d_model = d_model

        # Input
        # event, pitch, duration, velocity, track_number, instrument, tempo, position, measure_number
        embedding_vocabs_size = [
            vocabs['events']['size'],
            vocabs['pitchs']['size'],
            vocabs['durations']['size'],
            vocabs['velocities']['size'],
            vocabs['track_numbers']['size'],
            vocabs['instruments']['size'],
            vocabs['tempos']['size'],
            vocabs['positions']['size'],
            vocabs['max_seq_length']
        ]
        self._embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=d_model,
                padding_idx=self._pad_id # [...] the embedding vector at padding_idx will default to all zeros [...]
            )
            for vsize in embedding_vocabs_size
        ])
        # Output
        # event, pitch, duration, velocity, track_number, (instrument), (position)
        logit_vocabs_size = [
            vocabs['events']['size'],
            vocabs['pitchs']['size'],
            vocabs['durations']['size'],
            vocabs['velocities']['size'],
            vocabs['track_numbers']['size'],
            vocabs['instruments']['size'], # NOTE: should we predict instrument?
            vocabs['positions']['size'],
        ]
        if vocabs['paras']['position_method'] == 'event':
            logit_vocabs_size.pop()
        self._logits = nn.ModuleList([
            nn.Linear(
                in_features=d_model,
                out_features=vsize
            )
            for vsize in logit_vocabs_size[:-1]
        ])

        layer = nn.TransformerEncoderLayer( # name's encoder, used as decoder. Cause we don't need memory
            d_model=d_model,
            nhead=attn_heads_number,
            batch_first=True
        )
        self._transformer_decoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=layers_number
        )

    def forward(self, x, mask):
        # x.shape = (batch, sequence, feature)
        print(len(self._embeddings))
        x = sum(
            emb(x[:,:,i]) for i, emb in enumerate(self._embeddings)
        )
        x = self._transformer_decoder(
            src=x,
            mask=mask
        )
        out = [
            logit(x) for logit in self._logits
        ]
        return out

    def calc_loss(self, pred_logit, target):
        """
            pred is a list
            - length=out_feature_num
            - elements are tenors with shape=(batch_size, seq_size, vocabs_size_of_feature)
            target has shape: (batch_size, seq_size, out_feature_num)
        """
        # basically treat seq_size as the K in
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
        head_losses = [
            cross_entropy(
                input=pred_feature_logit.transpose(1, 2), # (batch_size, vocabs_size_of_feature, seq_size)
                target=target[..., i] # (batch_size, seq_size)
            )
            for i, pred_feature_logit in enumerate(pred_logit)
        ]
        loss = sum(head_losses)
        return loss


    def calc_set_loss(self, pred_logit, target):
        """
            pred has shape: (batch_size, seq_size, out_feature_num, vocabs_size)
            target has shape: (batch_size, seq_size, mps_length, 1)
        """
        pass


def get_seq_mask(size: int):
    return triu(ones(size, size) * float('-inf'), diagonal=1)
