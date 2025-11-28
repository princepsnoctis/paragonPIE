import torch
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            path_to_csv,

            name_converter,
            unit_converter,
            tax_converter
    ):
        super().__init__()

        # Read data
        df = pd.read_csv(path_to_csv, na_values=[""])

        # Extract samples
        self.samples = []

        for sample in df["sample"]:
            sample = name_converter.encode_seq(sample)

            self.samples.append(sample)

        # Extract labels and process them completely (here not in __getitem__, since the dataset is so tiny, roughly 70KB), the __getitem__'s job will be to only convert data to tensors
        regression_columns = ["amount", "quantity", "price", "total"]

        self.labels = []

        for _, row in df.iterrows():
            label = {
                # Sequential
                "name": [name_converter["<BOS>"]] + (name_converter.encode_seq(row["name"]) if pd.notna(row["name"]) else [name_converter["<NONE>"]]) + [name_converter["<EOS>"]],

                # Categorical
                "unit": unit_converter[row["unit"]] if pd.notna(row["unit"]) else unit_converter["<NONE>"],
                "tax":  tax_converter [row["tax"] ] if pd.notna(row["tax"] ) else tax_converter ["<NONE>"],

                # Numerical
                **{col             : row[col] if pd.notna(row[col]) else -1 for col in regression_columns}, # Cannot be None since tensor requires a single dtype, the -1 will be masked out anyway
                **{col + "_present": pd.notna(row[col])                     for col in regression_columns}  # Will indicate if the information is even present in the sample, will be used for masking and for inference because for non-present data model will just spit out random numbers
            }

            self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Sample
        sample = self.samples[idx]
        sample = torch.tensor(sample, dtype=torch.long)

        # Label
        regression_columns = ["amount", "quantity", "price", "total"]

        label = self.labels[idx]
        label = {
            "name": torch.tensor(label["name"], dtype=torch.long),
            "unit": torch.tensor(label["unit"], dtype=torch.long),
            "tax":  torch.tensor(label["tax"],  dtype=torch.long),

            **{col             : torch.tensor(label[col]             , dtype=torch.float32) for col in regression_columns},
            **{col + "_present": torch.tensor(label[col + "_present"], dtype=torch.float32)    for col in regression_columns}
        }

        return sample, label