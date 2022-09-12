import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .corpus import FEATURE_INDEX, INPUT_FEATURE_NAME, OUTPUT_FEATURE_NAME, Vocabs, array_to_text_list, text_list_to_array
from .midi import piece_to_midi
from .tokens import PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR

class MidiTransformerDecoder(nn.Module):
    def __init__(self,
            vocabs: Vocabs,
            max_seq_length: int,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[PADDING_TOKEN_STR] == 0

        self.vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim

        # Input features
        self.embedding_vocabs_size = [
            ( getattr(vocabs, feature_name).size if feature_name != 'measure_numbers' else max_seq_length )
            for feature_name in INPUT_FEATURE_NAME
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vsize,
                embedding_dim=embedding_dim,
                padding_idx=vocabs.events.text2id[PADDING_TOKEN_STR]
                # [...] the embedding vector at padding_idx will default to all zeros [...]
            )
            for vsize in self.embedding_vocabs_size
        ])
        # Output features
        self.logit_vocabs_size = [
            getattr(vocabs, feature_name).size for feature_name in OUTPUT_FEATURE_NAME
        ]
        if vocabs.paras['position_method'] == 'event':
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
        except Exception as e:
            for i, vsize in enumerate(self.embedding_vocabs_size):
                if torch.all(0 <= x[:,:,i]) and torch.all(x[:,:,i] < vsize):
                    print(i, True)
                else:
                    print(i)
                    for b in range(x.shape[0]):
                        if not torch.all(0 <= x[b,:,i]) and torch.all(x[b,:,i] < vsize):
                            print(x[b,:,i])
            raise e

        x = self.transformer_decoder(
            src=x,
            mask=mask
        )
        logits = [
            logit(x) for logit in self.logits
        ]
        return logits

def generate_sample(model: MidiTransformerDecoder, steps: int, start_seq: torch.Tensor = None, temperature=1.0) -> list:
    """
        start_seq is Tensor with shape: (1, seq_size, in_feature_number)
        - if start_seq is None, will use `text_list_to_array([BEGIN_TOKEN_STR])` as start_seq
        return the text list of the generated piece
    """
    if start_seq is not None:
        if len(start_seq.shape) != 3:
            raise ValueError(f'start_seq\'s shape have to be (1, seq_length, input_feature_num), get {start_seq.shape}')
        elif start_seq.shape[0] != 1 or start_seq.shape[2] != len(INPUT_FEATURE_NAME):
            raise ValueError(f'start_seq\'s shape have to be (1, seq_length, input_feature_num), get {start_seq.shape}')
    else:
        start_seq = torch.from_numpy(text_list_to_array([BEGIN_TOKEN_STR], model.vocabs)).unsqueeze(0).int()
    max_length_mask = get_seq_mask(model.max_seq_length).to(start_seq.device)

    # get model device
    device = next(model.parameters()).device
    start_seq.to(device)
    training_state = model.training
    model.eval()

    output_text_list = array_to_text_list(start_seq[0].cpu().numpy(), vocabs=model.vocabs, is_input=True)

    input_seq = start_seq
    output_seq = model.to_output_features(start_seq)
    end_with_end_token = False
    # print(output_seq.shape)
    # for _ in tqdm(range(steps)):
    for _ in range(steps):
        clipped_input_seq = input_seq[:, -model.max_seq_length:]
        mask = max_length_mask[:clipped_input_seq.shape[1], :clipped_input_seq.shape[1]]
        logits = model.forward(clipped_input_seq, mask)
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
            new_output_text_list = array_to_text_list(new_output_seq[0].cpu().numpy(), vocabs=model.vocabs, is_input=False)
            # print(new_output_text_list)

            try:
                # format checking
                piece_to_midi(' '.join(new_output_text_list), model.vocabs.paras['nth'])
                # format checking and position, tempo, measure_number calculation
                input_seq = torch.from_numpy(
                    text_list_to_array(new_output_text_list, vocabs=model.vocabs)
                ).unsqueeze(0).long()
            except:
                try_count += 1
                continue # keep sampling until no error

            output_seq = new_output_seq
            output_text_list = new_output_text_list
            break

        if try_count == try_count_limit:
            break

        if sampled_features[FEATURE_INDEX['evt']] == model.vocabs.events.text2id[END_TOKEN_STR]:
            end_with_end_token = True
            break

    if not end_with_end_token:
        output_text_list.append(END_TOKEN_STR)

    model.train(training_state)
    return output_text_list


def calc_losses(pred_logit, target_logit):
    """
        pred_logit is a list
        - length: out_feature_num
        - elements are tenors with shape: (batch_size, seq_size, feature_vocab_size)
        target_logit has shape: (batch_size, seq_size, out_feature_num)
    """
    # basically treat seq_size as the K in
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
    # target_logit have to be long
    target_logit = target_logit.long()
    head_losses = [
        F.cross_entropy(
            input=pred_feature_logit.transpose(1, 2), # (batch_size, feature_vocab_size, seq_size)
            target=target_logit[..., i] # (batch_size, seq_size)
        )
        for i, pred_feature_logit in enumerate(pred_logit)
    ]
    return head_losses
    # loss = sum(head_losses)
    # return loss

def calc_permutable_subseq_losses(pred_logit, target_logit, batched_mps_indices):
    """
        pred_logit is a list
        - length: out_feature_num
        - elements are tenors with shape: (batch_size, seq_size, feature_vocabs_size)
        target_logit has shape: (batch_size, seq_size, out_feature_num)
        mps_indices is a numpy object array of numpy int16 arrays in varying lengths
    """
    target_logit = target_logit.long()
    min_head_losses_list = []
    for batch_number, mps_indices in enumerate(batched_mps_indices):
        for mps_number in enumerate(mps_indices.shape[0] - 1):
            end_index = mps_indices[mps_number+1]
            for begin_index in range(mps_indices[mps_number], end_index):
                all_head_losses = []
                for seq_index in range(begin_index, end_index):
                    all_head_losses.append([
                        F.cross_entropy(
                            input=pred_feature_logit[batch_number, seq_index], # shape = (feature_vocabs_size, )
                            target=target_logit[batch_number, seq_index, k] # shape = ()
                        )
                        for k, pred_feature_logit in enumerate(pred_logit)
                    ])
                min_head_losses = min(all_head_losses, key=sum)
                min_head_losses_list.append(min_head_losses)
    # swap axis: from (batch_size*seq_size, out_feature_num) to (out_feature_num, batch_size*seq_size)
    min_head_losses_list = zip(*min_head_losses_list)
    head_losses = [sum(losses) for losses in min_head_losses_list]
    return head_losses


def get_seq_mask(size: int):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()


# def save_model_meta_info(model: MidiTransformerDecoder, save_dir_path: str):
#     meta_info_dict = {
#         'vocabs': model.vocabs.to_dict(),
#         'max_seq_length': model.max_seq_length,
#         'layers_number': model.layers_number,
#         'attn_heads_number': model.attn_heads_number,
#         'embedding_dim': model.embedding_dim
#     }
#     with open(os.path.join(save_dir_path, 'model_meta_info.json'), 'w+', encoding='utf8') as meta_info_file:
#         json.dump(meta_info_dict, meta_info_file)


# def load_model_meta_info()
