#!/usr/bin/env python3
"""
Extract the seventh sheet from an Excel file and create:
1. A CSV file with protein intensities (proteins as rows, samples as columns)
2. A label file where UPS1 proteins = 1, yeast proteins = 0
"""

import pandas as pd
import argparse
import os


def extract_and_transform(excel_file, sheet_index=6):
    """
    Extract sheet from Excel and create intensity matrix and labels.
    
    Args:
        excel_file: Path to Excel file
        sheet_index: 0-indexed sheet number (6 = seventh sheet)
    
    Returns:
        df: DataFrame with intensities (proteins as rows, samples as columns)
    """
    print(f"Reading Excel file: {excel_file}")
    print(f"Extracting sheet index {sheet_index} (sheet #{sheet_index + 1})")
    
    # Read the specified sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_index)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean up protein IDs before setting as index
    # Handle cases like "P06396ups;CON__Q3SX14" -> "P06396ups"
    # Handle cases like ">O76070ups|SYUG_HUMAN_UPS..." -> "O76070ups"
    def clean_protein_id(protein_id):
        protein_str = str(protein_id)
        # Remove leading ">" if present
        if protein_str.startswith('>'):
            protein_str = protein_str[1:]
        # Split by ";" and take first part
        if ';' in protein_str:
            protein_str = protein_str.split(';')[0]
        # Split by "|" and take first part
        if '|' in protein_str:
            protein_str = protein_str.split('|')[0]
        return protein_str.strip()
    
    df['Majority protein IDs'] = df['Majority protein IDs'].apply(clean_protein_id)
    
    # Set "Majority protein IDs" as index
    df = df.set_index('Majority protein IDs')
    
    # Select only the log intensity columns
    log_columns = ['A1 (log)', 'A2 (log)', 'A3 (log)', 'B1 (log)', 'B2 (log)', 'B3 (log)']
    df_intensities = df[log_columns].copy()
    
    # Rename columns to remove " (log)" suffix
    df_intensities.columns = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    
    # Create label column based on protein IDs
    # UPS1 proteins typically have "UPS" or "ups" in their ID
    labels = []
    for protein_id in df_intensities.index:
        protein_str = str(protein_id).upper()
        # Label as 1 if it contains UPS, 0 otherwise (yeast)
        if 'UPS' in protein_str:
            labels.append(1)
        else:
            labels.append(0)
    
    # Add labels as column
    df_intensities['is_differentially_expressed'] = labels
    
    print(f"\nData shape: {df_intensities.shape}")
    print(f"UPS1 proteins (label=1): {sum(labels)}")
    print(f"Yeast proteins (label=0): {len(labels) - sum(labels)}")
    
    return df_intensities


def make_sample_labels(df):
    """Create sample labels from sample names."""
    # For proteomics data, we'll create labels based on sample names
    # This can be customized based on your sample naming convention
    sample_labels = {}
    for sample in df.columns:
        # Skip the label column
        if sample == 'is_differentially_expressed':
            continue
        # Extract group from sample name (e.g., A0 -> A, B0 -> B)
        if len(str(sample)) > 0:
            group = str(sample)[0]
            # Assign label based on group
            sample_labels[sample] = 1 if group == 'A' else 0
        else:
            sample_labels[sample] = 0
    
    sample_labels_df = pd.DataFrame.from_dict(
        sample_labels, orient='index', columns=['label']
    )
    return sample_labels_df


if __name__ == "__main__":
    # Setup Argument Parser for OmniBenchmark
    parser = argparse.ArgumentParser(
        description="Extract Excel sheet and create proteomics dataset"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        required=True,
        help="Dataset name"
    )
    parser.add_argument(
        "--excel_file",
        type=str,
        default="raw_ss_dataset.xlsx",
        help="Path to Excel file (relative to script location)"
    )
    parser.add_argument(
        "--sheet",
        type=int,
        default=6,
        help="Sheet index (0-based, default: 6 for seventh sheet)"
    )
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full path to Excel file
    # If excel_file is relative, join with script directory
    if not os.path.isabs(args.excel_file):
        excel_path = os.path.join(script_dir, args.excel_file)
    else:
        excel_path = args.excel_file
    
    print(f"Looking for Excel file at: {excel_path}")
    
    # Extract data from Excel
    df = extract_and_transform(excel_path, args.sheet)
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct output file paths
    output_file = os.path.join(args.output_dir, f"{args.name}.dataset.csv")
    protein_labels_file = os.path.join(args.output_dir, f"{args.name}.true_labels_proteins.csv")
    sample_labels_file = os.path.join(args.output_dir, f"{args.name}.true_labels.csv")
    
    # Print summary to console
    print(f"\nData Extracted Successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save main dataset to CSV (without labels column, no index name header)
    df_data = df.drop(columns=['is_differentially_expressed'])
    df_data.index.name = None  # Remove index name to avoid header for first column
    df_data.to_csv(output_file)
    print(f"\nSaved dataset to '{output_file}'")
    
    # Save protein labels separately (protein_id, label - no header)
    protein_labels = df['is_differentially_expressed']
    protein_labels.to_csv(protein_labels_file, header=False)
    print(f"Saved true protein labels to '{protein_labels_file}'")
    
    # Save sample labels separately (sample_id, label - no header)
    sample_labels = make_sample_labels(df_data)
    sample_labels.to_csv(sample_labels_file, header=False)
    print(f"Saved true sample labels to '{sample_labels_file}'")
