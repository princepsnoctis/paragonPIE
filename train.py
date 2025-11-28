import torch

from collator import Collator
from converter import Converter
from dataset import Dataset
from model import Model

from utils import all_unique_characters_in_csv



PATH_TO_DATA_CSV = "data/data.csv"
MAX_SAM_LEN = 128
MAX_NAM_LEN = 128
BATCH_SIZE = 256
EPOCHS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

# Dataset, collator, DataLoader
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
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

# Build model
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
).to(device)

model.load_state_dict(torch.load("model4.pt"))

# Loss functions
def seq_loss(logits, targets, pad_idx):
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_idx,
    )

def reg_loss(preds, present, targets):
    if present.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    present = present.bool()

    present_preds   = preds[present]
    present_targets = targets[present]

    present_preds = torch.clamp(present_preds, min=0.0)

    # These require positive values !!!
    present_preds_log   = torch.log1p(present_preds)
    present_targets_log = torch.log1p(present_targets)

    loss = torch.nn.functional.mse_loss(
        present_preds_log.reshape(-1),
        present_targets_log
    )

    return loss

def cat_loss(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets)

def bin_loss(logits, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1),
        targets
    )

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

for epoch in range(EPOCHS):
    batch_loss = 0

    for samples, labels, sample_padding_mask, name_padding_mask in dataloader:
        samples = samples.to(device)
        sample_padding_mask = sample_padding_mask.to(device)
        name_padding_mask = name_padding_mask.to(device)

        labels_names = labels["names"].to(device)

        labels_units = labels["units"].to(device)
        labels_taxes = labels["taxes"].to(device)

        labels_amounts  = labels["amounts"].to(device)
        labels_quantity = labels["quantities"].to(device)
        labels_price    = labels["prices"].to(device)
        labels_total    = labels["totals"].to(device)

        labels_amount_present   = labels["amount_presents"].to(device)
        labels_quantity_present = labels["quantity_presents"].to(device)
        labels_price_present    = labels["price_presents"].to(device)
        labels_total_present    = labels["total_presents"].to(device)

        # Forward pass
        outputs = model(
            encoder_tokens=samples,
            decoder_tokens=labels_names[:, :-1],
            sample_mask=sample_padding_mask,
            name_mask=name_padding_mask[:, :-1]
        )

        # print("###############################################")
        # print("Sample: ", name_converter.decode_seq(samples[0].tolist()))
        # print("Decoder input: ", name_converter.decode_seq(labels_names[:, :-1][0].tolist()))
        # print("Decoder output: ", name_converter.decode_seq(labels_names[:, 1:][0].tolist()))
        # break

        name_loss = seq_loss(outputs["name_logits"], labels_names[:, 1:], name_converter["<PAD>"])

        unit_loss             = cat_loss(outputs["unit_logits"], labels_units)
        tax_loss              = cat_loss(outputs["tax_logits"], labels_taxes)

        amount_loss           = reg_loss(outputs["amount_pred"], labels_amount_present, labels_amounts)
        quantity_loss         = reg_loss(outputs["quantity_pred"], labels_quantity_present, labels_quantity)
        price_loss            = reg_loss(outputs["price_pred"], labels_price_present, labels_price)
        total_loss            = reg_loss(outputs["total_pred"], labels_total_present, labels_total)

        amount_present_loss   = bin_loss(outputs["amount_present_logit"], labels_amount_present)
        quantity_present_loss = bin_loss(outputs["quantity_present_logit"], labels_quantity_present)
        price_present_loss    = bin_loss(outputs["price_present_logit"], labels_price_present)
        total_present_loss    = bin_loss(outputs["total_present_logit"], labels_total_present)

        # print("###########################################################")
        # print("name_loss:", name_loss)
        # print("unit_loss:", unit_loss)
        # print("tax_loss:", tax_loss)
        # print("amount_loss:", amount_loss)
        # print("quantity_loss:", quantity_loss)
        # print("price_loss:", price_loss)
        # print("total_loss:", total_loss)
        # print("amount_present_loss:", amount_present_loss)
        # print("quantity_present_loss:", quantity_present_loss)
        # print("price_present_loss:", price_present_loss)
        # print("total_present_loss:", total_present_loss)

        loss = (
            1 * name_loss +
            1 * unit_loss +
            1 * tax_loss +
            1 * amount_loss +
            1 * quantity_loss +
            1 * price_loss +
            1 * total_loss +
            1 * amount_present_loss +
            1 * quantity_present_loss +
            1 * price_present_loss +
            1 * total_present_loss
        )

        optimizer.zero_grad() # Before backward so losses across batches don't accumulate

        # Backward pass
        loss.backward()

        optimizer.step()

        # Measurement stuff
        batch_loss += loss.item()

    avg_loss = batch_loss / len(dataloader)

    print(f"EPOCH {epoch}: loss={avg_loss:.6f}")

torch.save(model.state_dict(), "model5.pt")