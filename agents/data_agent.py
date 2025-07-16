import pandas as pd
import os

class DataAgent:
    def fetch_data(self, path="data/raw/housing.csv"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found at {path}")
        return pd.read_csv(path)

    def clean_data(self, df):
        df = df.dropna()
        df = df[df["price"] > 0]
        df["age"] = 2025 - df["year_built"]
        df = pd.get_dummies(df, columns=["property_type", "location"], drop_first=True)
        return df