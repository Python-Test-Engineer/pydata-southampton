"""
Titanic Dataset Load Module

This module handles saving processed data and generating data quality reports.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

class TitanicDataLoader:
    """Class to handle loading/saving processed Titanic data."""
    
    def __init__(self, output_dir: str = "../../data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_processed_data(self, df: pd.DataFrame, filename: str = "titanic_processed.csv") -> str:
        """
        Save processed dataset to CSV file.
        
        Args:
            df (pd.DataFrame): Processed dataset
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        try:
            output_path = self.output_dir / filename
            df.to_csv(output_path, index=False)
            console.print(f"[green]✓ Processed data saved to {output_path}[/green]")
            return str(output_path)
        except Exception as e:
            console.print(f"[red]✗ Error saving processed data: {str(e)}[/red]")
            raise
    
    def save_feature_sets(self, df: pd.DataFrame) -> dict:
        """
        Save different feature sets for various analysis purposes.
        
        Args:
            df (pd.DataFrame): Processed dataset
            
        Returns:
            dict: Paths to saved feature sets
        """
        console.print("[cyan]Saving feature sets...[/cyan]")
        
        saved_files = {}
        
        # Basic features for simple analysis
        basic_features = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        basic_features = [col for col in basic_features if col in df.columns]
        if basic_features:
            basic_df = df[basic_features]
            saved_files['basic'] = self.save_processed_data(basic_df, "titanic_basic_features.csv")
        
        # Engineered features for advanced analysis
        engineered_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['family_size', 'is_alone', 'age_group', 'fare_per_person', 'title', 'has_cabin'])]
        if engineered_cols:
            engineered_features = basic_features + engineered_cols
            engineered_features = [col for col in engineered_features if col in df.columns]
            engineered_df = df[engineered_features]
            saved_files['engineered'] = self.save_processed_data(engineered_df, "titanic_engineered_features.csv")
        
        # ML-ready features (encoded and scaled)
        ml_features = [col for col in df.columns if col != 'survived']  # All features except target
        ml_df = df[['survived'] + ml_features]
        saved_files['ml_ready'] = self.save_processed_data(ml_df, "titanic_ml_ready.csv")
        
        console.print(f"[green]✓ Saved {len(saved_files)} feature sets[/green]")
        return saved_files
    
    def generate_data_dictionary(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive data dictionary.
        
        Args:
            df (pd.DataFrame): Processed dataset
            
        Returns:
            dict: Data dictionary
        """
        console.print("[cyan]Generating data dictionary...[/cyan]")
        
        data_dict = {
            'dataset_info': {
                'name': 'Titanic Dataset - Processed',
                'shape': df.shape,
                'created_date': datetime.now().isoformat(),
                'total_features': len(df.columns),
                'total_records': len(df)
            },
            'features': {}
        }
        
        for column in df.columns:
            col_info = {
                'data_type': str(df[column].dtype),
                'non_null_count': int(df[column].count()),
                'null_count': int(df[column].isnull().sum()),
                'null_percentage': round(df[column].isnull().sum() / len(df) * 100, 2)
            }
            
            if df[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[column].min()) if pd.notna(df[column].min()) else None,
                    'max': float(df[column].max()) if pd.notna(df[column].max()) else None,
                    'mean': float(df[column].mean()) if pd.notna(df[column].mean()) else None,
                    'std': float(df[column].std()) if pd.notna(df[column].std()) else None,
                    'median': float(df[column].median()) if pd.notna(df[column].median()) else None
                })
            else:
                col_info.update({
                    'unique_values': int(df[column].nunique()),
                    'most_frequent': str(df[column].mode().iloc[0]) if len(df[column].mode()) > 0 else None,
                    'sample_values': df[column].dropna().unique()[:5].tolist()
                })
            
            data_dict['features'][column] = col_info
        
        return data_dict
    
    def save_data_dictionary(self, data_dict: dict, filename: str = "data_dictionary.json") -> str:
        """
        Save data dictionary to JSON file.
        
        Args:
            data_dict (dict): Data dictionary
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(data_dict, f, indent=2, default=str)
            console.print(f"[green]✓ Data dictionary saved to {output_path}[/green]")
            return str(output_path)
        except Exception as e:
            console.print(f"[red]✗ Error saving data dictionary: {str(e)}[/red]")
            raise
    
    def generate_quality_report(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> dict:
        """
        Generate a data quality report comparing original and processed data.
        
        Args:
            df_original (pd.DataFrame): Original dataset
            df_processed (pd.DataFrame): Processed dataset
            
        Returns:
            dict: Quality report
        """
        console.print("[cyan]Generating data quality report...[/cyan]")
        
        quality_report = {
            'processing_summary': {
                'original_records': len(df_original),
                'processed_records': len(df_processed),
                'records_removed': len(df_original) - len(df_processed),
                'original_features': len(df_original.columns),
                'processed_features': len(df_processed.columns),
                'features_added': len(df_processed.columns) - len(df_original.columns)
            },
            'missing_data_analysis': {
                'original_missing_total': int(df_original.isnull().sum().sum()),
                'processed_missing_total': int(df_processed.isnull().sum().sum()),
                'missing_data_reduction': int(df_original.isnull().sum().sum() - df_processed.isnull().sum().sum())
            },
            'feature_analysis': {
                'original_features': list(df_original.columns),
                'processed_features': list(df_processed.columns),
                'new_features': [col for col in df_processed.columns if col not in df_original.columns],
                'removed_features': [col for col in df_original.columns if col not in df_processed.columns]
            },
            'data_types': {
                'numerical_features': len(df_processed.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(df_processed.select_dtypes(include=['object', 'category']).columns),
                'boolean_features': len(df_processed.select_dtypes(include=['bool']).columns)
            }
        }
        
        # Calculate completeness score
        total_cells_original = df_original.shape[0] * df_original.shape[1]
        total_cells_processed = df_processed.shape[0] * df_processed.shape[1]
        
        completeness_original = (total_cells_original - df_original.isnull().sum().sum()) / total_cells_original
        completeness_processed = (total_cells_processed - df_processed.isnull().sum().sum()) / total_cells_processed
        
        quality_report['quality_metrics'] = {
            'original_completeness': round(completeness_original * 100, 2),
            'processed_completeness': round(completeness_processed * 100, 2),
            'completeness_improvement': round((completeness_processed - completeness_original) * 100, 2)
        }
        
        return quality_report
    
    def save_quality_report(self, quality_report: dict, filename: str = "quality_report.json") -> str:
        """
        Save quality report to JSON file.
        
        Args:
            quality_report (dict): Quality report
            filename (str): Output filename
            
        Returns:
            str: Path to saved file
        """
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(quality_report, f, indent=2, default=str)
            console.print(f"[green]✓ Quality report saved to {output_path}[/green]")
            return str(output_path)
        except Exception as e:
            console.print(f"[red]✗ Error saving quality report: {str(e)}[/red]")
            raise
    
    def load_and_save_all(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> dict:
        """
        Complete loading process: save data, generate and save reports.
        
        Args:
            df_original (pd.DataFrame): Original dataset
            df_processed (pd.DataFrame): Processed dataset
            
        Returns:
            dict: Summary of all saved files
        """
        console.print("[bold blue]Starting complete data loading process...[/bold blue]")
        
        saved_files = {}
        
        # Save feature sets
        feature_sets = self.save_feature_sets(df_processed)
        saved_files.update(feature_sets)
        
        # Generate and save data dictionary
        data_dict = self.generate_data_dictionary(df_processed)
        saved_files['data_dictionary'] = self.save_data_dictionary(data_dict)
        
        # Generate and save quality report
        quality_report = self.generate_quality_report(df_original, df_processed)
        saved_files['quality_report'] = self.save_quality_report(quality_report)
        
        console.print(f"[bold green]✓ Loading process completed! Saved {len(saved_files)} files[/bold green]")
        
        # Print summary
        console.print("\n[bold yellow]Files Saved:[/bold yellow]")
        for file_type, path in saved_files.items():
            console.print(f"  {file_type}: {path}")
        
        return saved_files

def main():
    """Main function to demonstrate loading capabilities."""
    console.print("[bold blue]Titanic Data Loading Module[/bold blue]")
    console.print("This module provides data saving and quality reporting capabilities.")

if __name__ == "__main__":
    main()
