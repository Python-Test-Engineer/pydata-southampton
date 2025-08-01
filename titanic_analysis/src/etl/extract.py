"""
Titanic Dataset Extraction Module

This module handles downloading and loading the Titanic dataset from seaborn.
"""

import pandas as pd
import seaborn as sns
import os
from pathlib import Path
from rich.console import Console

console = Console()

def download_titanic_data(save_path: str = None) -> pd.DataFrame:
    """
    Download the Titanic dataset from seaborn and optionally save it.
    
    Args:
        save_path (str, optional): Path to save the raw dataset
        
    Returns:
        pd.DataFrame: Raw Titanic dataset
    """
    try:
        console.print("[cyan]Downloading Titanic dataset from seaborn...[/cyan]")
        
        # Load the Titanic dataset from seaborn
        titanic_df = sns.load_dataset('titanic')
        
        console.print(f"[green]✓ Successfully loaded Titanic dataset with {len(titanic_df)} rows and {len(titanic_df.columns)} columns[/green]")
        
        # Save to file if path is provided
        if save_path:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            titanic_df.to_csv(save_path, index=False)
            console.print(f"[green]✓ Dataset saved to {save_path}[/green]")
        
        return titanic_df
        
    except Exception as e:
        console.print(f"[red]✗ Error downloading dataset: {str(e)}[/red]")
        raise

def load_titanic_data(file_path: str) -> pd.DataFrame:
    """
    Load Titanic dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        console.print(f"[cyan]Loading Titanic dataset from {file_path}...[/cyan]")
        df = pd.read_csv(file_path)
        console.print(f"[green]✓ Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns[/green]")
        
        return df
        
    except Exception as e:
        console.print(f"[red]✗ Error loading dataset: {str(e)}[/red]")
        raise

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the Titanic dataset structure and return basic information.
    
    Args:
        df (pd.DataFrame): Dataset to validate
        
    Returns:
        dict: Dataset validation information
    """
    expected_columns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    
    validation_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_expected_columns': [col for col in expected_columns if col not in df.columns.str.lower()],
        'data_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    console.print("[yellow]Dataset Validation Summary:[/yellow]")
    console.print(f"Shape: {validation_info['shape']}")
    console.print(f"Columns: {validation_info['columns']}")
    console.print(f"Missing values: {sum(validation_info['missing_values'].values())} total")
    console.print(f"Duplicate rows: {validation_info['duplicate_rows']}")
    
    if validation_info['missing_expected_columns']:
        console.print(f"[red]⚠ Missing expected columns: {validation_info['missing_expected_columns']}[/red]")
    else:
        console.print("[green]✓ All expected columns present[/green]")
    
    return validation_info

def main():
    """Main function to extract Titanic data."""
    # Define paths
    raw_data_path = "../../data/raw/titanic.csv"
    
    try:
        # Download and save the dataset
        df = download_titanic_data(raw_data_path)
        
        # Validate the dataset
        validation_info = validate_dataset(df)
        
        # Display basic info
        console.print("\n[bold blue]Dataset Preview:[/bold blue]")
        console.print(df.head().to_string())
        
        console.print("\n[bold blue]Dataset Info:[/bold blue]")
        console.print(df.info())
        
        return df, validation_info
        
    except Exception as e:
        console.print(f"[red]✗ Extraction failed: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    main()
