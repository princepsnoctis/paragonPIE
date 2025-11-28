import pandas as pd

df = pd.read_csv("./data.csv")

# df["total"] = df["total"].str.replace(",", ".").astype(float)
#
# df.to_csv("./data.csv", index=False)