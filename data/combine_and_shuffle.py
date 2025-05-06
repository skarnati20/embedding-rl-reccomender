import pandas as pd
import numpy as np
import os
import argparse


def combine_and_shuffle_csvs(csv_files, output_file):
    """
    Combine multiple CSV files with identical columns and shuffle the entries.
    Preserves string formatting for all values.

    Parameters:
    -----------
    csv_files : list
        List of paths to CSV files to combine
    output_file : str
        Path to save the combined and shuffled CSV file
    """
    # Check if all files exist
    for file in csv_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

    # Read and combine all CSV files with string preservation
    print(f"Combining {len(csv_files)} CSV files...")

    # Use converters parameter to treat all columns as strings
    combined_dfs = []
    for file in csv_files:
        # First read the header to identify columns
        header = pd.read_csv(file, nrows=0).columns.tolist()
        # Create a converter dict to force string type for all columns
        converters = {col: str for col in header}
        # Read with converters to preserve string format
        df = pd.read_csv(file, converters=converters)
        combined_dfs.append(df)

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Get the total number of rows
    total_rows = len(combined_df)
    print(f"Total rows: {total_rows}")

    # Shuffle the combined DataFrame
    print("Shuffling data...")
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save the combined and shuffled data
    # Use quoting parameters to ensure all fields are quoted
    print(f"Saving combined and shuffled data to {output_file}...")
    combined_df.to_csv(
        output_file, index=False, quoting=1
    )  # quoting=1 is csv.QUOTE_ALL
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine and shuffle multiple CSV files with identical columns"
    )
    parser.add_argument("files", nargs="+", help="CSV files to combine")
    parser.add_argument(
        "-o", "--output", default="combined_shuffled.csv", help="Output file name"
    )

    args = parser.parse_args()

    combine_and_shuffle_csvs(args.files, args.output)
