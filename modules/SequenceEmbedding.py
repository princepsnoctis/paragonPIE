import torch
import torch.nn as nn


class SequenceEmbedding(nn.Module):
    """
    Sequence embedding layer used for embedding tokens and their positions in a sequence.

    :param vocabulary_size: Vocabulary size
    :type  vocabulary_size: int

    :param max_sequence_length: Maximum length of sequence
    :type  max_sequence_length: int

    :param embedding_dimension: Embedding dimension
    :type  embedding_dimension: int
    """
    def __init__(self, vocabulary_size, max_sequence_length, embedding_dimension=512):
        """

        :param vocabulary_size: Vocabulary size
        :type  vocabulary_size: int

        :param max_sequence_length: Maximum length of sequence
        :type  max_sequence_length: int

        :param embedding_dimension: Embedding dimension
        :type  embedding_dimension: int
        """
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = max_sequence_length

        self.token_embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dimension)
        self.positional_embedding = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=embedding_dimension)

    def forward(self, token_ids):
        """
        Forward pass through the module.

        :param token_ids: token IDs
        :type  token_ids: torch.Tensor

        :return: Token and positional embeddings: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :rtype:  torch.Tensor
        """
        B, max_sequence_length = token_ids.shape

        positions = torch.arange(max_sequence_length, device=token_ids.device).unsqueeze(0)  # (1, L)

        token_embedding = self.token_embedding(token_ids)  # (B, L, D)
        positional_embedding = self.positional_embedding(positions)  # (1, L, D)

        out = token_embedding + positional_embedding

        return out
