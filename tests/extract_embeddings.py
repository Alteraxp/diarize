import torch
import numpy as np
from pathlib import Path
import wespeakerruntime as wespeaker_rt

# init model
model = wespeaker_rt.Speaker(lang="en")

def enroll(audio_path, output_path):
    emb = model.extract_embedding(audio_path)

    if emb is None:
        raise ValueError(f"Failed to extract embedding from {audio_path}")

    if emb.ndim == 2:
        emb = emb[0]

    emb = emb.astype(np.float32)

    # normalize (important)
    emb = emb / np.linalg.norm(emb)

    torch.save(torch.from_numpy(emb), output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    enroll("pruthvi.wav", "../embeddings/Pruthvi.pt")
    enroll("alice.wav", "../embeddings/Alice.pt")
