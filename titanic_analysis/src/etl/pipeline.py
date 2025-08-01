"""
Titanic ETL Pipeline

Main pipeline that orchestrates the complete ETL process for Titanic dataset.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from etl.extract import download_titanic_data, validate_dataset
from etl.transform import TitanicTransformer
from etl.load import TitanicDataLoader
from rich.console import Console
import pandas as pd

console = Console()

class TitanicETLPipeline:
    """Complete ETL pipeline for Titanic dataset."""
    
    def __init__(self, data_dir: str = "../../data"):
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Initialize components
        self.transformer = TitanicTransformer()
        self.loader = TitanicDataLoader(str(self.processed_data_dir))
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self) -> pd.DataFrame:
        """
        Extract phase: Download and load the Titanic dataset.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        console.print("[bold blue]🚢 EXTRACT PHASE[/bold blue]")
        
        raw_data_path = self.raw_data_dir / "titanic.csv"
        
        try:
            # Download the dataset
            df_raw = download_titanic_data(str(raw_data_path))
            
            # Validate the dataset
            validation_info = validate_dataset(df_raw)
            
            console.print(f"[green]✓ Extract phase completed successfully[/green]")
            console.print(f"[yellow]Dataset shape: {df_raw.shape}[/yellow]")
            
            return df_raw
            
        except Exception as e:
            console.print(f"[red]✗ Extract phase failed: {str(e)}[/red]")
            raise
    
    def transform(self, df_raw: pd.DataFrame, scale_features: bool = True) -> pd.DataFrame:
        """
        Transform phase: Clean data and engineer features.
        
        Args:
            df_raw (pd.DataFrame): Raw dataset
            scale_features (bool): Whether to scale numerical features
            
        Returns:
            pd.DataFrame: Transformed dataset
        """
        console.print("[bold blue]⚙️ TRANSFORM PHASE[/bold blue]")
        
        try:
            # Apply all transformations
            df_transformed = self.transformer.transform(df_raw, scale_features=scale_features)
            
            # Generate transformation summary
            summary = self.transformer.get_feature_summary(df_raw, df_transformed)
            
            console.print(f"[green]✓ Transform phase completed successfully[/green]")
            console.print(f"[yellow]Transformed dataset shape: {df_transformed.shape}[/yellow]")
            
            return df_transformed
            
        except Exception as e:
            console.print(f"[red]✗ Transform phase failed: {str(e)}[/red]")
            raise
    
    def load(self, df_raw: pd.DataFrame, df_transformed: pd.DataFrame) -> dict:
        """
        Load phase: Save processed data and generate reports.
        
        Args:
            df_raw (pd.DataFrame): Raw dataset
            df_transformed (pd.DataFrame): Transformed dataset
            
        Returns:
            dict: Summary of saved files
        """
        console.print("[bold blue]💾 LOAD PHASE[/bold blue]")
        
        try:
            # Save all data and generate reports
            saved_files = self.loader.load_and_save_all(df_raw, df_transformed)
            
            console.print(f"[green]✓ Load phase completed successfully[/green]")
            
            return saved_files
            
        except Exception as e:
            console.print(f"[red]✗ Load phase failed: {str(e)}[/red]")
            raise
    
    def run_pipeline(self, scale_features: bool = True) -> dict:
        """
        Run the complete ETL pipeline.
        
        Args:
            scale_features (bool): Whether to scale numerical features
            
        Returns:
            dict: Pipeline execution summary
        """
        console.print("[bold magenta]🚀 STARTING TITANIC ETL PIPELINE[/bold magenta]")
        console.print("=" * 60)
        
        pipeline_summary = {
            'status': 'running',
            'phases_completed': [],
            'saved_files': {},
            'errors': []
        }
        
        try:
            # Phase 1: Extract
            df_raw = self.extract()
            pipeline_summary['phases_completed'].append('extract')
            pipeline_summary['raw_data_shape'] = df_raw.shape
            
            console.print("\n" + "=" * 60)
            
            # Phase 2: Transform
            df_transformed = self.transform(df_raw, scale_features=scale_features)
            pipeline_summary['phases_completed'].append('transform')
            pipeline_summary['transformed_data_shape'] = df_transformed.shape
            
            console.print("\n" + "=" * 60)
            
            # Phase 3: Load
            saved_files = self.load(df_raw, df_transformed)
            pipeline_summary['phases_completed'].append('load')
            pipeline_summary['saved_files'] = saved_files
            
            # Pipeline completed successfully
            pipeline_summary['status'] = 'completed'
            
            console.print("\n" + "=" * 60)
            console.print("[bold green]🎉 ETL PIPELINE COMPLETED SUCCESSFULLY![/bold green]")
            console.print(f"[yellow]Raw data: {df_raw.shape}[/yellow]")
            console.print(f"[yellow]Processed data: {df_transformed.shape}[/yellow]")
            console.print(f"[yellow]Files saved: {len(saved_files)}[/yellow]")
            
            return pipeline_summary
            
        except Exception as e:
            pipeline_summary['status'] = 'failed'
            pipeline_summary['errors'].append(str(e))
            console.print(f"[bold red]💥 PIPELINE FAILED: {str(e)}[/bold red]")
            raise
    
    def get_pipeline_status(self) -> dict:
        """
        Get the current status of pipeline components.
        
        Returns:
            dict: Pipeline status information
        """
        status = {
            'directories': {
                'raw_data_dir': str(self.raw_data_dir),
                'processed_data_dir': str(self.processed_data_dir),
                'raw_data_exists': self.raw_data_dir.exists(),
                'processed_data_exists': self.processed_data_dir.exists()
            },
            'components': {
                'transformer_initialized': self.transformer is not None,
                'loader_initialized': self.loader is not None
            }
        }
        
        # Check for existing files
        raw_files = list(self.raw_data_dir.glob("*.csv")) if self.raw_data_dir.exists() else []
        processed_files = list(self.processed_data_dir.glob("*")) if self.processed_data_dir.exists() else []
        
        status['files'] = {
            'raw_files': [f.name for f in raw_files],
            'processed_files': [f.name for f in processed_files]
        }
        
        return status

def main():
    """Main function to run the ETL pipeline."""
    try:
        # Initialize and run the pipeline
        pipeline = TitanicETLPipeline()
        
        # Display pipeline status
        status = pipeline.get_pipeline_status()
        console.print("[bold blue]Pipeline Status:[/bold blue]")
        console.print(f"Raw data directory: {status['directories']['raw_data_dir']}")
        console.print(f"Processed data directory: {status['directories']['processed_data_dir']}")
        
        # Run the complete pipeline
        summary = pipeline.run_pipeline(scale_features=True)
        
        console.print("\n[bold blue]Pipeline Summary:[/bold blue]")
        console.print(f"Status: {summary['status']}")
        console.print(f"Phases completed: {summary['phases_completed']}")
        console.print(f"Files saved: {len(summary.get('saved_files', {}))}")
        
        return summary
        
    except Exception as e:
        console.print(f"[bold red]Pipeline execution failed: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
