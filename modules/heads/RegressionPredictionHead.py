import torch.nn as nn


class RegressionPredictionHead(nn.Module):
    """
    Regression prediction head used for continuous-value prediction in a multitask transformer model.
    This module processes the sequence encoder output using the first token as a sequence representation.

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
        Forward pass of the module.

        :param encoder_output: Output of the preceding sequence encoder: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :type  encoder_output: torch.Tensor

        :return: Continuous-value prediction: Tensor of shape (batch_size, 1)
        :rtype:  torch.Tensor
        """
        first_token = encoder_output[:, 0, :]

        pred = self.mlp(first_token)

        pred = nn.functional.softplus(pred)  # Ensure all values are non-negative for MSLE loss

        return pred
