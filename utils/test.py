import torch

from collator import Collator
from converter import Converter
from dataset import Dataset

from utils import all_unique_characters_in_csv

PATH_TO_DATA_CSV = "data/data.csv"

# Converters
name_converter = Converter(
    symbols=all_unique_characters_in_csv(PATH_TO_DATA_CSV),
    special_symbols=["<BOS>", "<EOS>", "<NONE>", "<PAD>"],
)

unit_converter = Converter(
    symbols=["SZT", "G", "KG", "L", "ML"],
    special_symbols=["<NONE>"],
)

tax_converter = Converter(
    symbols=["A", "B", "C"],
    special_symbols=["<NONE>"],
)

dataset = Dataset(
    path_to_csv=PATH_TO_DATA_CSV,
    name_converter=name_converter,
    unit_converter=unit_converter,
    tax_converter=tax_converter,
)

collator = Collator(
    name_converter=name_converter,
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
)

for samples, labels, sample_padding_masks, name_padding_masks in dataloader:
    print(labels)

    break