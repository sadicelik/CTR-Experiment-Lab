import pandas as pd

if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Device": ["PC", "laptop", "mobile", "tablet"],
        }
    )
    print(df)

    # Apply one-hot encoding
    encoded_df = pd.get_dummies(df, columns=["Device"], dtype=int)
    print(encoded_df)
