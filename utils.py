import pandas as pd

def all_unique_characters_in_csv(path_to_csv):
    df = pd.read_csv(path_to_csv, dtype=str)

    df_as_text = df.astype(str).agg("".join, axis=1).str.cat()

    return sorted(set(df_as_text))

def all_unique_column_values(path_to_csv, column):
    df = pd.read_csv(path_to_csv)

    filtered = df[column].dropna()
    filtered = filtered[filtered != ""]

    return filtered.unique().tolist()