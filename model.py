import torch
import torch.nn as nn

import modules
import modules.heads

class Model(nn.Module):
    def __init__(
        self,
        sym_len,      # vocabulary size

        # These are for embeddings
        max_sam_len,  # max size of sample (input)
        max_nam_len,  # max size of name (output)

        # There are for heads
        unit_cat_len,
        tax_cat_len,

        # Hyperparams just for the model modules
        emb_dim=512,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        num_layers=4
    ):
        super().__init__()

        self.encoder_embedding = modules.SequenceEmbedding(
            vocabulary_size=sym_len,
            max_sequence_length=max_sam_len,
            embedding_dimension=emb_dim
        )

        self.decoder_embedding = modules.SequenceEmbedding(
            vocabulary_size=sym_len,
            max_sequence_length=max_nam_len, # max_sam_len and max_nam_len --- do we keep then different? does it matter?
            embedding_dimension=emb_dim
        )

        self.encoder = modules.SequenceEncoder(
            embedding_dimension=emb_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        self.decoder = modules.SequenceDecoder(
            embedding_dimension=emb_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_layers
        )

        self.multihead = modules.heads.MultitaskPredictionHead(
            name_vocabulary_size=sym_len,
            unit_number_of_categories=unit_cat_len,
            tax_number_of_categories=tax_cat_len,
            embedding_dimension=emb_dim
        )

    def forward(
            self,
            encoder_tokens,
            decoder_tokens,
            sample_mask,
            name_mask
    ):
        encoder_embedding = self.encoder_embedding(encoder_tokens)
        decoder_embedding = self.decoder_embedding(decoder_tokens)

        encoder_output = self.encoder(encoder_embedding, sample_mask)

        decoder_output = self.decoder(
            decoder_input=decoder_embedding,
            encoder_output=encoder_output,
            input_padding_mask=sample_mask,
            output_padding_mask=name_mask,
        )

        outputs = self.multihead(
            encoder_output=encoder_output,
            decoder_output=decoder_output,
        )

        return outputs

    @torch.no_grad()
    def generate_name(
            self,
            encoder_input,
            bos_idx,
            eos_idx,
            max_nam_len,
    ):
        # Get encoding once
        encoder_input_embedding = self.encoder_embedding(encoder_input)

        sample_mask = torch.zeros_like(encoder_input).bool()

        encoder_output = self.encoder(encoder_input_embedding, sample_mask)

        # Generate sequence
        sequence = [bos_idx]

        for i in range(max_nam_len):
            decoder_input = torch.tensor(sequence).unsqueeze(0) # BOS

            decoder_input_embedding = self.decoder_embedding(decoder_input)

            name_mask = torch.zeros_like(decoder_input).bool()

            decoder_output = self.decoder(
                decoder_input=decoder_input_embedding,
                encoder_output=encoder_output,
                name_mask=name_mask,
                sample_mask=sample_mask,
            )

            logits = self.multihead.head_name(decoder_output)

            last_token_logits = logits[:, -1, :]

            last_token_probs = torch.nn.functional.softmax(last_token_logits, dim=-1)

            next_token = last_token_probs.argmax(dim=-1).item()

            sequence.append(next_token)

            if next_token == eos_idx:
                break

        return sequence