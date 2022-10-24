import torch
from torch import nn
import torch.nn.functional as F

from .corpus import TOKEN_ATTR_INDEX, COMPLETE_ATTR_NAME, OUTPUT_ATTR_NAME, Vocabs, array_to_text_list, text_list_to_array
from .midi import piece_to_midi
from .tokens import PADDING_TOKEN_STR, BEGIN_TOKEN_STR, END_TOKEN_STR

class MyMidiTransformer(nn.Module):
    def __init__(self,
            vocabs: Vocabs,
            max_seq_length: int,
            layers_number: int,
            attn_heads_number: int,
            embedding_dim: int,
            input_no_tempo: bool,
            input_no_time_signature: bool
            ) -> None:
        super().__init__()

        assert vocabs.events.text2id[PADDING_TOKEN_STR] == 0

        self.vocabs = vocabs
        self.max_seq_length = max_seq_length
        self.layers_number = layers_number
        self.attn_heads_number = attn_heads_number
        self.embedding_dim = embedding_dim
        self.input_no_tempo = input_no_tempo
        self.input_no_time_signature = input_no_time_signature

        # Input attributess

        # the indices of input attrs in complete attribute list
        self.input_attrs_indices = [
            TOKEN_ATTR_INDEX[fname]
            for fname in COMPLETE_ATTR_NAME
            if not (input_no_tempo and fname=='tempos') or not (input_no_time_signature and fname=='time_signatures')
        ]
        self.embedding_vocabs_size = [
            ( getattr(vocabs, fname).size if fname != 'measure_numbers' else max_seq_length )
            for fname in COMPLETE_ATTR_NAME
            if not (input_no_tempo and fname=='tempos') or not (input_no_time_signature and fname=='time_signatures')
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

        # Output attributes
        self.output_attrs_indices = [
            TOKEN_ATTR_INDEX[fname]
            for fname in OUTPUT_ATTR_NAME
        ]
        # because position is the last attribute in OUTPUT_ATTR_NAME
        if vocabs.paras['position_method'] == 'event':
            self.output_attrs_indices.pop()
        self.logit_vocabs_size = [
            getattr(vocabs, OUTPUT_ATTR_NAME[i]).size
            for i in self.output_attrs_indices
        ]
        self.logits = nn.ModuleList([
            nn.Linear(
                in_features=embedding_dim,
                out_features=vsize
            )
            for vsize in self.logit_vocabs_size
        ])

        layer = nn.TransformerEncoderLayer( # name's encoder, used as decoder.
            d_model=embedding_dim,
            nhead=attn_heads_number,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=layers_number
        )

    # batched_seq_complete_attrs has shape: (batch_size, seq_size, complete_attr_num)
    def to_input_attrs(self, batched_seq_complete_attrs):
        return batched_seq_complete_attrs[..., self.input_attrs_indices]

    def to_output_attrs(self, batched_seq_complete_attrs):
        return batched_seq_complete_attrs[..., self.output_attrs_indices]

    def forward(self, x, mask):
        # x has shape: (batch_size, seq_size, in_attr_number)
        x = sum(
            emb(x[:,:,i]) for i, emb in enumerate(self.embeddings)
        )
        # try:
        #     x = sum(
        #         emb(x[:,:,i]) for i, emb in enumerate(self.embeddings)
        #     )
        # except Exception as e:
        #     for i, vsize in enumerate(self.embedding_vocabs_size):
        #         if torch.all(0 <= x[:,:,i]) and torch.all(x[:,:,i] < vsize):
        #             print(i, True)
        #         else:
        #             print(i)
        #             for b in range(x.shape[0]):
        #                 if not torch.all(0 <= x[b,:,i]) and torch.all(x[b,:,i] < vsize):
        #                     print(x[b,:,i])
        #     raise e
        x = self.transformer_decoder(
            src=x,
            mask=mask
        )
        logits = tuple(
            logit(x) for logit in self.logits
        )
        return logits

def generate_sample(model: MyMidiTransformer, steps: int, start_seq = None, temperature=1.0, try_count_limit=1000, print_exception=False) -> list:
    """
        Expect start_seq to be Tensor with shape: (1, seq_size, complete_attr_number) or None
        - if start_seq is None, will use `text_list_to_array([BEGIN_TOKEN_STR])` as start_seq
        return the text list of the generated piece
    """
    if start_seq is not None:
        exception_msg = f'start_seq\'s shape have to be (1, seq_length, complete_attr_number), get {start_seq.shape}'
        if len(start_seq.shape) != 3:
            raise AssertionError(exception_msg)
        elif start_seq.shape[0] != 1 or start_seq.shape[2] != len(COMPLETE_ATTR_NAME):
            raise AssertionError(exception_msg)
    else:
        start_seq = torch.from_numpy(text_list_to_array([BEGIN_TOKEN_STR], model.vocabs)).unsqueeze(0).int()
    max_length_mask = get_seq_mask(model.max_seq_length).to(start_seq.device)

    # get model device
    device = next(model.parameters()).device
    start_seq.to(device)
    training_state = model.training
    model.eval()

    output_text_list = array_to_text_list(start_seq[0].cpu().numpy(), vocabs=model.vocabs, is_output=False)

    input_seq = start_seq
    output_seq = model.to_output_attrs(start_seq)
    end_with_end_token = False
    # print(output_seq.shape)
    # for _ in tqdm(range(steps)):
    for _ in range(steps):
        clipped_input_seq = input_seq[:, -model.max_seq_length:]
        mask = max_length_mask[:clipped_input_seq.shape[1], :clipped_input_seq.shape[1]]
        logits = model.forward(clipped_input_seq, mask)
        last_logits = [
            l[:, -1, :] # l has shape (1, sequence, attr_vocab_size)
            for l in logits
        ]
        try_count = 0
        while try_count < try_count_limit:
            sampled_attrs = [
                torch.multinomial(F.softmax(l[0] / temperature, dim=0), 1)
                for l in last_logits # l has shape (1, attr_vocab_size)
            ]
            new_token = torch.stack(sampled_attrs, dim=1) # shape = (1, attr_num)
            # print([ model.vocabs.to_dict()[OUTPUT_FEATURE_NAME[i]]['id2text'][int(f)] for i, f in enumerate(new_token[0]) ])
            new_token = torch.unsqueeze(new_token, dim=0) # shape = (1, 1, attr_num)
            new_output_seq = torch.cat((output_seq, new_token), dim=1)
            new_output_text_list = array_to_text_list(new_output_seq[0].cpu().numpy(), vocabs=model.vocabs, is_output=True)
            # print(new_output_text_list)
            if new_output_text_list[-1] == END_TOKEN_STR:
                # if sampled EOS, then just end
                break

            try:
                # format checking
                # have to append EOS at the end to not raise error
                piece_to_midi(' '.join(new_output_text_list + [END_TOKEN_STR]), model.vocabs.paras['nth'])
                # format checking and position, tempo, measure_number calculation
                input_seq = torch.from_numpy(
                    text_list_to_array(new_output_text_list + [END_TOKEN_STR], vocabs=model.vocabs)
                ).unsqueeze(0).long()
            except Exception as e:
                if print_exception:
                    print(repr(e))
                try_count += 1
                continue # keep sampling until no error

            output_seq = new_output_seq
            output_text_list = new_output_text_list
            break

        if try_count == try_count_limit:
            break

        if new_output_text_list[-1] == END_TOKEN_STR:
            end_with_end_token = True
            break

    if not end_with_end_token:
        output_text_list.append(END_TOKEN_STR)

    model.train(training_state)
    return output_text_list


def calc_losses(pred_logit, target_logit):
    """
        pred_logit is a list
        - length: out_attr_num
        - elements are tenors with shape: (batch_size, seq_size, attr_vocab_size)
        target_logit has shape: (batch_size, seq_size, out_attr_num)
        return a list of losses of each head
    """
    # basically treat seq_size as one of the K dimensions and K = 1
    # target_logit have to be long int
    target_logit = target_logit.long()
    head_losses = [
        F.cross_entropy(
            input=pred_attr_logit.transpose(1, 2), # (batch_size, attr_vocab_size, seq_size)
            target=target_logit[..., k] # (batch_size, seq_size)
        )
        for k, pred_attr_logit in enumerate(pred_logit)
    ]
    return head_losses
    # loss = sum(head_losses)
    # return loss

def calc_permutable_subseq_losses(pred_logit, target_logit, batched_mps_indices):
    """
        pred_logit is a list
        - length: out_attr_num
        - elements are tenors with shape: (batch_size, seq_size, attr_vocabs_size)
        target_logit has shape: (batch_size, seq_size, out_attr_num)
        mps_indices is a numpy object array of numpy int16 arrays in varying lengths
        return a list of losses of each head
    """
    target_logit = target_logit.long()
    # min_head_losses = [[] for _ in range(len(pred_logit))]
    for batch_number, mps_indices in enumerate(batched_mps_indices):
        min_head_losses_indices = []
        mps_indices_with_begin_and_end = [0] + [m for m in mps_indices] + [target_logit.shape[1]]
        for i in range(len(mps_indices_with_begin_and_end) - 1):
            end_index = mps_indices_with_begin_and_end[i+1]
            for begin_index in range(mps_indices_with_begin_and_end[i], end_index):
                mps_size = end_index-begin_index
                begin_index_pred_attr_logit = [
                    pred_attr_logit[batch_number, begin_index].unsqueeze(0).expand((mps_size, -1))
                    # expand into (1, attr_vocabs_size) and then expand into (mps_size, attr_vocabs_size)
                    # shape = (mps_size, attr_vocabs_size)
                    for pred_attr_logit in pred_logit
                ]
                begin_index_head_losses = [
                    F.cross_entropy(
                        input=begin_index_pred_attr_logit[k], # shape = (mps_size, attr_vocabs_size)
                        target=target_logit[batch_number, begin_index:end_index, k], # shape = (mps_size)
                        reduction='none' # return (mps_size, )
                    )
                    for k, _ in enumerate(pred_logit)
                ]
                # head_losses is now a list of out_attr_num tensors, each has shape (mps_size, )
                min_head_losses_arg = min(
                    range(mps_size),
                    key=lambda x: sum([l[x] for l in begin_index_head_losses])
                )
                # print(begin_index, ': mps_size', mps_size, 'min_head_losses_arg', min_head_losses_arg)
                min_head_losses_indices.append(
                    begin_index + min_head_losses_arg
                )
                # for k, _ in enumerate(min_head_losses):
                #     min_head_losses[k].append(begin_index_head_losses[k][min_head_losses_arg])
            # end for begin_index
        # end for mps_number
        # modify target logit such that the target at ith index is now the target at min_head_losses_indices[i]
        for i, min_head_losses_index in enumerate(min_head_losses_indices):
            target_logit[batch_number, i] = target_logit[batch_number, min_head_losses_index]
    head_losses = calc_losses(pred_logit, target_logit)
    # compare
    # print([float(hl) for hl in head_losses])
    # print([
    #     float(sum(mhl) / len(mhl))
    #     for mhl in min_head_losses
    # ])
    return head_losses


def get_seq_mask(size: int):
    return torch.triu(torch.ones(size, size), diagonal=1).bool()

MidiTransformerDecoder = MyMidiTransformer # old name
