import pandas as pd
import torch
import torch.nn as nn

if __name__ == "__main__":
    torch.manual_seed(1773)  # Set seed

    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Device": ["PC", "laptop", "mobile", "tablet"],
            "Con_Type": ["LAN", "Wi-Fi", "LTE", "5G"],
        }
    )

    # Encode categorical features as integer IDs
    device2id = {cat: idx for idx, cat in enumerate(df["Device"].unique())}
    con2id = {cat: idx for idx, cat in enumerate(df["Con_Type"].unique())}
    df["Device_ID"] = df["Device"].map(device2id)
    df["Con_Type_ID"] = df["Con_Type"].map(con2id)
    print(df)

    # Embedding Layers
    device_emb = nn.Embedding(len(device2id), 3)  # (4,3)
    con_emb = nn.Embedding(len(con2id), 2)  # (4,2)
    print(f"{device_emb = }")
    print(f"{con_emb = }")

    device_ids = torch.tensor(df["Device_ID"].values)  # torch.Size([4])
    con_ids = torch.tensor(df["Con_Type_ID"].values)  # torch.Size([4])

    # Look up embeddings
    device_vecs = device_emb(device_ids)  # shape: (4, 3)
    con_vecs = con_emb(con_ids)  # shape: (4, 2)

    # Concatenate embeddings into final input
    final_vecs = torch.cat([device_vecs, con_vecs], dim=1)

    print(f"Device embeddings:\n", device_vecs)
    print(f"Con_Type embeddings:\n", con_vecs)
    print(f"Final concatenated vectors:\n", final_vecs)
