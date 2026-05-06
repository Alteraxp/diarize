import torch
import numpy as np
from pathlib import Path
import argparse
import wespeakerruntime as wespeaker_rt

# init model
model = wespeaker_rt.Speaker(lang="en")


def enroll(audio_path: Path, output_path: Path):
    emb = model.extract_embedding(str(audio_path))

    if emb is None:
        raise ValueError(f"Failed to extract embedding from {audio_path}")

    if emb.ndim == 2:
        emb = emb[0]

    emb = emb.astype(np.float32)

    # normalize (important)
    emb = emb / np.linalg.norm(emb)

    torch.save(torch.from_numpy(emb), output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enroll speaker embeddings from WAV files")

    parser.add_argument(
        "input",
        type=str,
        help="Path to input .wav file OR directory containing .wav files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save embeddings"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        if input_path.suffix.lower() != ".wav":
            raise ValueError("Input file must be a .wav file")

        output_path = output_dir / (input_path.stem + ".pt")
        enroll(input_path, output_path)

    elif input_path.is_dir():
        wav_files = list(input_path.glob("*.wav"))

        if not wav_files:
            raise ValueError("No .wav files found in directory")

        for wav_file in wav_files:
            output_path = output_dir / (wav_file.stem + ".pt")
            enroll(wav_file, output_path)

    else:
        raise ValueError("Invalid input path")


if __name__ == "__main__":
    main()
