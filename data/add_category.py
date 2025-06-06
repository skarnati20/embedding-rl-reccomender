import os
import argparse
import pandas as pd


def add_category_column(input_csv, output_csv=None):
    """
    Add a category column to a CSV file based on the CSV filename.

    Args:
        input_csv (str): Path to the input CSV file
        output_csv (str, optional): Path to output CSV file. If None, overwrites input file.
    """
    # Extract category from input CSV filename (without extension)
    category = os.path.splitext(os.path.basename(input_csv))[0]
    print(f"Adding category column with value: '{category}'")

    # Read the CSV file
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Add the category column
    df["category"] = category

    # Determine output file
    if output_csv is None:
        output_csv = input_csv
        print(f"Overwriting original file...")
    else:
        print(f"Saving to {output_csv}...")

    # Save the modified CSV
    df.to_csv(output_csv, index=False)
    print(f"Done! Added 'category' column with value '{category}' to {len(df)} rows.")


def main():
    parser = argparse.ArgumentParser(
        description="Add a category column to a CSV file based on the filename"
    )
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (optional, defaults to overwriting input file)",
    )

    args = parser.parse_args()

    add_category_column(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
