import torch.nn as nn


class BinaryPredictionHead(nn.Module):
    """
    Binary prediction head used for predicting binary states of attributes presences.
    This module processes the sequence encode output using the first token as a sequence representation.

    :param embedding_dimension: The embedding dimension.
    :type  embedding_dimension: int
    """
    def __init__(self, embedding_dimension=512):
        """
        Initializes the module.

        :param embedding_dimension: The embedding dimension.
        :type  embedding_dimension: int
        """

        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dimension, out_features=embedding_dimension),
            nn.ReLU(),
            nn.Linear(in_features=embedding_dimension, out_features=1)
        )

    def forward(self, encoder_output):
        """
        Forward pass through the module.

        :param encoder_output: Output from the preceding encoder: Tensor of shape (batch_size, max_sequence_length, embedding_dimension).
        :type  encoder_output: torch.Tensor

        :return: Logits (pre-sigmoid scores): Tensor of shape (batch_size, 1).
        :rtype:  torch.Tensor
        """
        first_token = encoder_output[:, 0, :]

        logit = self.mlp(first_token)

        return logit
