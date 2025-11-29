import torch.nn as nn


class SequenceEncoder(nn.Module):
    """
    Sequence encoder module used for encoding sequences, it leverages a transformer encoder.
    This module's output is later used and processed by several prediction heads and a decoder.

    :param embedding_dimension: The embedding dimension.
    :type  embedding_dimension: int

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
        :type  embedding_dimension: int

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # More stable training
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # For norm first
        )

        self.final_norm = nn.LayerNorm(embedding_dimension)

    def forward(self, encoder_input, sample_padding_mask):
        """
        Forward pass of the module.

        :param encoder_input: Embedded token ids: Tensor of shape (batch_size, max_sequence_length, embedding_dimension).
        :type  encoder_input: torch.Tensor

        :param sample_padding_mask: A mask to ignore padding tokens, True at the padding token positions: Tensor of shape (batch_size, max_sequence_length).
        :type  sample_padding_mask: torch.Tensor

        :return: The encoded sequence: Tensor of shape (batch_size, max_sequence_length, embedding_dimension).
        :rtype:  torch.Tensor
        """
        out = self.encoder(encoder_input, src_key_padding_mask=sample_padding_mask)
        out = self.final_norm(out)

        return out
