import pandas as pd

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Device": ["PC", "laptop", "mobile", "tablet"],
            "Con_Type": ["LAN", "Wi-Fi", "LTE", "5G"],
        }
    )
    print(df)

    # Apply one-hot encoding
    encoded_df = pd.get_dummies(df, columns=["Device", "Con_Type"], dtype=int)
    print(encoded_df)
