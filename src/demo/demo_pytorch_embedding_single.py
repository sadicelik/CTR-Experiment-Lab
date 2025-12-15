import pandas as pd
import torch
import torch.nn as nn

if __name__ == "__main__":
    torch.manual_seed(1773)  # Set seed

    df = pd.DataFrame(
        {"ID": [1, 2, 3, 4], "Device": ["PC", "laptop", "mobile", "tablet"]}
    )

    # Encode categorical feature as integer ID
    device2id = {cat: idx for idx, cat in enumerate(df["Device"].unique())}
    df["Device_ID"] = df["Device"].map(device2id)
    print(df)

    # Embedding Layer
    device_emb = nn.Embedding(
        num_embeddings=len(device2id), embedding_dim=2
    )  # Embedding(4,2)
    print(f"{device_emb = }")

    device_ids = torch.tensor(df["Device_ID"].values)  # torch.Size([4])
    print(f"{device_ids = }")
    print(f"{device_ids.size() = }")

    device_vecs = device_emb(device_ids)  # (4, 2)
    print(f"Device embeddings:\n", device_vecs)

    # Print embeddings separately
    for device, idx in device2id.items():
        emb_vec = device_emb(torch.tensor(idx))
        print(f"Embedding for {device}: {emb_vec.detach().numpy()}")
