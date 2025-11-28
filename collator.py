import torch

class Collator:
    def __init__(self, name_converter):
        self.name_converter = name_converter

    def __call__(self, batch):
        # Only samples and names in labels must be padded, they are the only sequential data
        samples, labels = map(list, zip(*batch))

        names             = [label["name"]     for label in labels]
        units             = [label["unit"]     for label in labels]
        taxes             = [label["tax"]      for label in labels]
        amounts           = [label["amount"]   for label in labels]
        quantities        = [label["quantity"] for label in labels]
        prices            = [label["price"]    for label in labels]
        totals            = [label["total"]    for label in labels]
        amount_presents   = [label["amount_present"]    for label in labels]
        quantity_presents = [label["quantity_present"]    for label in labels]
        price_presents    = [label["price_present"]    for label in labels]
        total_presents    = [label["total_present"]    for label in labels]

        padded_samples = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=self.name_converter["<PAD>"])
        padded_names   = torch.nn.utils.rnn.pad_sequence(names,   batch_first=True, padding_value=self.name_converter["<PAD>"])

        samples_padding_mask = (padded_samples == self.name_converter["<PAD>"])
        names_padding_mask   = (padded_names   == self.name_converter["<PAD>"])

        padded_labels = {
            "names"            : padded_names,

            "units"            : torch.stack(units),
            "taxes"            : torch.stack(taxes),

            "amounts"          : torch.stack(amounts),
            "quantities"       : torch.stack(quantities),
            "prices"           : torch.stack(prices),
            "totals"           : torch.stack(totals),

            "amount_presents"  : torch.stack(amount_presents),
            "quantity_presents": torch.stack(quantity_presents),
            "price_presents"   : torch.stack(price_presents),
            "total_presents"   : torch.stack(total_presents),
        }

        return padded_samples, padded_labels, samples_padding_mask, names_padding_mask