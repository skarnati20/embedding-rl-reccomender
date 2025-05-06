import os
import csv
import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm


def process_text_files(input_dir, output_csv):
    """
    Process all text files in a directory and create a CSV with filename
    and text embedding.

    Args:
        input_dir (str): Path to the directory containing text files
        output_csv (str): Path to output CSV file
    """
    # Initialize the sentence transformer model for embeddings
    print("Loading the embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    data = []

    # Get all files in the directory (including subdirectories)
    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    print(f"Found {len(all_files)} files. Processing...")

    # Process all files
    for file_path in tqdm(all_files):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Generate embedding for the entire content
            embedding = model.encode(content)

            # Add to our data list
            data.append(
                {
                    "filename": os.path.relpath(file_path, input_dir),
                    "embedding": embedding.tolist(),
                }
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(data)

    # Save to CSV
    print(f"Saving to {output_csv}...")
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Process text files into CSV with embeddings"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing text files",
    )
    parser.add_argument(
        "--output_csv", type=str, required=True, help="Path to output CSV file"
    )

    args = parser.parse_args()

    process_text_files(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
