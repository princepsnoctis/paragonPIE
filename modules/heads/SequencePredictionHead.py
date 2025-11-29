import torch.nn as nn


class SequencePredictionHead(nn.Module):
    """
    Sequence prediction heads used for predicting the succeeding token for each token in the decoder input.

    :param vocabulary_size: The vocabulary size
    :type  vocabulary_size: int

    :param embedding_dimension: The embedding dimension
    :type  vocabulary_size: int
    """
    def __init__(self, vocabulary_size, embedding_dimension=512):
        """
        Initializes the module.

        :param vocabulary_size: The vocabulary size
        :type  vocabulary_size: int

        :param embedding_dimension: The embedding dimension
        :type  embedding_dimension: int
        """
        super().__init__()

        self.linear = nn.Linear(in_features=embedding_dimension, out_features=vocabulary_size)

    def forward(self, decoder_output):
        """
        Forward pass through the module.

        :param decoder_output: Output of the preceding sequence decoder: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :type  decoder_output: torch.Tensor

        :return: Logits (pre-softmax scores): Tensor of shape (batch_size, max_sequence_length, vocabulary_size)
        :rtype:  torch.Tensor
        """
        logits = self.linear(decoder_output)

        return logits
