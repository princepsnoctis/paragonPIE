import torch
import torch.nn as nn


class SequenceDecoder(nn.Module):
    """
    Sequence decoder module used for decoding sequences, it leverages a transformer decoder.
    This module's output is later used and processed by sequence prediction head.

    :param embedding_dimension: The embedding dimension.
    :type   embedding_dimension: int

    :param num_heads: Number of attention heads.
    :type  num_heads: int

    :param ff_dim: Hidden dimension of the feedforward network.
    :type  ff_dim: int

    :param num_layers: Number of transformer blocks.
    :type  num_layers: int

    :param dropout: Dropout rate.
    :type  dropout: float
    """

    def __init__(self, embedding_dimension=512, num_heads=8, ff_dim=2048, num_layers=4, dropout=0.1):
        """
        Initializes the module.

        :param embedding_dimension: The embedding dimension.
        :type   embedding_dimension: int

        :param num_heads: Number of attention heads.
        :type  num_heads: int

        :param ff_dim: Hidden dimension of the feedforward network.
        :type  ff_dim: int

        :param num_layers: Number of transformer blocks.
        :type  num_layers: int

        :param dropout: Dropout rate.
        :type  dropout: float
        """
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dimension,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(embedding_dimension)

    def forward(self, decoder_input, encoder_output, input_padding_mask, output_padding_mask):
        """
        Forward pass of the module.

        :param decoder_input: Embedded token ids: Tensor of shape (batch_size, max_sequence_length, embedding_dimension).
        :type  decoder_input: torch.Tensor

        :param encoder_output: Sequence encoded by the sequence encoder: Tensor of shape (batch_size, max_sequence_length, embedding_dimension).
        :type  encoder_output: torch.Tensor

        :param input_padding_mask: A mask to ignore padding tokens in the input, True at the padding token positions: Tensor of shape (batch_size, max_sequence_length).
        :type  input_padding_mask: torch.Tensor

        :param output_padding_mask: A mask to ignore padding tokens in the output, True at the padding token positions: Tensor of shape (batch_size, max_sequence_length).
        :type  output_padding_mask: torch.Tensor

        :return: Normalized hidden states from the decoder (batch_size, max_sequence_length, embedding_dimension).
        :rtype:  torch.Tensor
        """
        name_seq_len = decoder_input.size(1)

        causal_mask = torch.triu(
            torch.ones((name_seq_len, name_seq_len), dtype=torch.bool, device=decoder_input.device), diagonal=1)

        x = self.decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=output_padding_mask,
            memory_key_padding_mask=input_padding_mask
        )

        x = self.final_norm(x)

        return x
