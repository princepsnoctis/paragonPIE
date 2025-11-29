import torch.nn as nn


class CategoryPredictionHead(nn.Module):
    """
    Category prediction head used for single-label classification tasks.
    This module processes the sequence encoder output using the first token as a sequence representation.

    :param number_of_categories: Number of categories.
    :type  number_of_categories: int

    :param embedding_dimension: The embedding dimension.
    :type  embedding_dimension: int
    """
    def __init__(self, number_of_categories, embedding_dimension=512):
        """
        Initializes the module.

        :param number_of_categories: Number of categories.
        :type  number_of_categories: int

        :param embedding_dimension: The embedding dimension.
        :type  embedding_dimension: int
        """

        super().__init__()

        self.linear = nn.Linear(in_features=embedding_dimension, out_features=number_of_categories)

    def forward(self, encoder_output):
        """
        Forward pass of the module.

        :param encoder_output: Output from the sequence encoder: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :type  encoder_output: torch.Tensor

        :return: Logits (pre-softmax scores): Tensor of shape (batch_size, number_of_categories)
        :rtype:  torch.Tensor
        """
        first_token = encoder_output[:, 0, :]

        logits = self.linear(first_token)

        return logits
