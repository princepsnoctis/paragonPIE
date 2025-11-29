import torch.nn as nn

import modules.heads


class MultitaskPredictionHead(nn.Module):
    """
    Multitask prediction head used for predicting multiple attribute types from a shared encoder output.
    This module bundles together several specialized heads responsible for predicting: `name`, `unit`, `tax`, `amount` and `quantity`.

    :param name_vocabulary_size: The amount of tokens in the vocabulary of name
    :type  name_vocabulary_size: int

    :param unit_number_of_categories: The amount of categories of the unit
    :type  unit_number_of_categories: int

    :param tax_number_of_categories: The amount of categories of the tax
    :type  tax_number_of_categories: int

    :param embedding_dimension: The embedding dimension
    :type  embedding_dimension: int
    """
    def __init__(self, name_vocabulary_size, unit_number_of_categories, tax_number_of_categories, embedding_dimension=512):
        """
        Initializes the module.

        :param name_vocabulary_size: The amount of tokens in the vocabulary of `name`
        :type  name_vocabulary_size: int

        :param unit_number_of_categories: The amount of categories of the `unit`
        :type  unit_number_of_categories: int

        :param tax_number_of_categories: The amount of categories of the `tax`
        :type  tax_number_of_categories: int

        :param embedding_dimension: The embedding dimension
        :type  embedding_dimension: int
        """
        super().__init__()

        self.head_name             = modules.heads.SequencePredictionHead(vocabulary_size=name_vocabulary_size, embedding_dimension=embedding_dimension)

        self.head_unit             = modules.heads.CategoryPredictionHead(number_of_categories=unit_number_of_categories, embedding_dimension=embedding_dimension)
        self.head_tax              = modules.heads.CategoryPredictionHead(number_of_categories=tax_number_of_categories, embedding_dimension=embedding_dimension)

        self.head_amount           = modules.heads.RegressionPredictionHead(embedding_dimension=embedding_dimension)
        self.head_quantity         = modules.heads.RegressionPredictionHead(embedding_dimension=embedding_dimension)
        self.head_price            = modules.heads.RegressionPredictionHead(embedding_dimension=embedding_dimension)
        self.head_total            = modules.heads.RegressionPredictionHead(embedding_dimension=embedding_dimension)

        self.head_amount_present   = modules.heads.BinaryPredictionHead(embedding_dimension=embedding_dimension)
        self.head_quantity_present = modules.heads.BinaryPredictionHead(embedding_dimension=embedding_dimension)
        self.head_price_present    = modules.heads.BinaryPredictionHead(embedding_dimension=embedding_dimension)
        self.head_total_present    = modules.heads.BinaryPredictionHead(embedding_dimension=embedding_dimension)

    def forward(self, encoder_output, decoder_output):
        """
        Forward pass through the module.

        :param encoder_output: Output of the preceding sequence encoder module: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :type  encoder_output: torch.Tensor

        :param decoder_output: Output of the preceding sequence decoder module: Tensor of shape (batch_size, max_sequence_length, embedding_dimension)
        :type  decoder_output: torch.Tensor

        :return: Dictionary containing outputs of all heads:
            - "logits--name":            Tensor of shape (batch_size, max_sequence_length, name_vocabulary_size)
            - "logits--unit":            Tensor of shape (batch_size, unit_number_of_categories)
            - "logits--tax":             Tensor of shape (batch_size, tax_number_of_categories)
            - "pred--amount":            Tensor of shape (batch_size, 1)
            - "pred--quantity":          Tensor of shape (batch_size, 1)
            - "pred--price":             Tensor of shape (batch_size, 1)
            - "pred--total":             Tensor of shape (batch_size, 1)
            - "present_logit--amount":   Tensor of shape (batch_size, 1)
            - "present_logit--quantity": Tensor of shape (batch_size, 1)
            - "present_logit--price":    Tensor of shape (batch_size, 1)
            - "present_logit--total":    Tensor of shape (batch_size, 1)
        :rtype:  dict
        """
        return {
            "logits--name"           : self.head_name(decoder_output),

            "logits--unit"           : self.head_unit(encoder_output),
            "logits--tax"            : self.head_tax(encoder_output),

            "pred--amount"           : self.head_amount(encoder_output),
            "pred--quantity"         : self.head_quantity(encoder_output),
            "pred--price"            : self.head_price(encoder_output),
            "pred--total"            : self.head_total(encoder_output),

            "present_logit--amount"  : self.head_amount_present(encoder_output),
            "present_logit--quantity": self.head_quantity_present(encoder_output),
            "present_logit--price"   : self.head_price_present(encoder_output),
            "present_logit--total"   : self.head_total_present(encoder_output),
        }
