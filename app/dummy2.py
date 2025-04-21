import pandas as pd
import os

def csv_to_filtered_json(csv_file_path, output_json_path=None):
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Failed to read the CSV file: {e}")
        return

    # Filter out rows where type == 'Income'
    df_filtered = df[df['type'] != 'Income']

    # Determine output file path
    if not output_json_path:
        base = os.path.splitext(csv_file_path)[0]
        output_json_path = base + "_expenses_only.json"

    # Save as JSON
    try:
        df_filtered.to_json(output_json_path, orient="records", lines=False, indent=4)
        print(f"Filtered JSON saved to: {output_json_path}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

# Example usage:
csv_to_filtered_json("../dataset/Sample_transactions.csv")
