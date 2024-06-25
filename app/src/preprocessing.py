import pandas as pd


cat_vals = ["регион", "использование", "pack"]
to_drop = ['client_id', 'mrg_',]


def import_data(path_to_file):
    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=to_drop, axis=1)
    return input_df


def run_preproc(input_df):
    input_df["регион"] = input_df["регион"].fillna("Nan")
    input_df["использование"] = input_df["использование"].fillna("Nan")
    input_df["pack"] = input_df["pack"].fillna("Nan")
    return input_df
