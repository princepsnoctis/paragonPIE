import torch

from converter import Converter
from model import Model

from utils import all_unique_characters_in_csv



PATH_TO_DATA_CSV = "data/data.csv"
MAX_NAM_LEN = 128
MAX_SAM_LEN = 128



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

# Model
model = Model(
    sym_len=len(name_converter),

    max_sam_len=MAX_NAM_LEN,
    max_nam_len=MAX_SAM_LEN,

    unit_cat_len=len(unit_converter),
    tax_cat_len=len(tax_converter),

    emb_dim=64,
    num_heads=1,
    ff_dim=128,
    dropout=0.3,
    num_layers=1
)

model.load_state_dict(torch.load("model.pt"))

model.eval()

sample = "OdbytySpermyWDupsku C 1SZT x1,33 1,33C"

encoded_sample = torch.tensor(
    name_converter.encode_seq(sample),
    dtype=torch.long,
).unsqueeze(0)

sequence = model.generate_name(
    encoder_input=encoded_sample,
    bos_idx=name_converter["<BOS>"],
    eos_idx=name_converter["<EOS>"],
    max_nam_len=MAX_NAM_LEN,
)

print("".join(name_converter.decode_seq(sequence)))