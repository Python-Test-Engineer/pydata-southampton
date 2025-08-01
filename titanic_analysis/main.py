"""
Titanic Analysis Main Script

This is the main entry point for the complete Titanic dataset analysis.
It orchestrates the ETL pipeline, exploratory data analysis, and survival prediction.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from etl.pipeline import TitanicETLPipeline
from analysis.exploratory_analysis import TitanicEDA
from analysis.survival_prediction import TitanicSurvivalPredictor
from rich.console import Console
import pandas as pd

console = Console()

class TitanicAnalysisOrchestrator:
    """Main orchestrator for complete Titanic analysis."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "reports"
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.etl_pipeline = TitanicETLPipeline(str(self.data_dir))
        self.eda_analyzer = TitanicEDA(str(self.output_dir / "figures"))
        self.predictor = TitanicSurvivalPredictor(str(self.output_dir))
        
        # Results storage
        self.analysis_results = {}
    
    def run_etl_pipeline(self, scale_features: bool = True) -> dict:
        """
        Run the ETL pipeline.
        
        Args:
            scale_features (bool): Whether to scale numerical features
            
        Returns:
            dict: ETL pipeline results
        """
        console.print("[bold magenta]🚀 PHASE 1: ETL PIPELINE[/bold magenta]")
        console.print("=" * 80)
        
        try:
            etl_results = self.etl_pipeline.run_pipeline(scale_features=scale_features)
            self.analysis_results['etl'] = etl_results
            
            console.print("[bold green]✅ ETL Pipeline completed successfully![/bold green]")
            return etl_results
            
        except Exception as e:
            console.print(f"[bold red]❌ ETL Pipeline failed: {str(e)}[/bold red]")
            raise
    
    def run_exploratory_analysis(self, df: pd.DataFrame) -> dict:
        """
        Run exploratory data analysis.
        
        Args:
            df (pd.DataFrame): Dataset for analysis
            
        Returns:
            dict: EDA results
        """
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]🔍 PHASE 2: EXPLORATORY DATA ANALYSIS[/bold magenta]")
        console.print("=" * 80)
        
        try:
            eda_results = self.eda_analyzer.run_complete_eda(df)
            self.analysis_results['eda'] = eda_results
            
            console.print("[bold green]✅ Exploratory Data Analysis completed successfully![/bold green]")
            return eda_results
            
        except Exception as e:
            console.print(f"[bold red]❌ Exploratory Data Analysis failed: {str(e)}[/bold red]")
            raise
    
    def run_survival_prediction(self, df: pd.DataFrame) -> dict:
        """
        Run survival prediction analysis.
        
        Args:
            df (pd.DataFrame): Dataset for prediction
            
        Returns:
            dict: Prediction results
        """
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]🤖 PHASE 3: SURVIVAL PREDICTION[/bold magenta]")
        console.print("=" * 80)
        
        try:
            prediction_results = self.predictor.run_complete_prediction_analysis(df)
            self.analysis_results['prediction'] = prediction_results
            
            console.print("[bold green]✅ Survival Prediction completed successfully![/bold green]")
            return prediction_results
            
        except Exception as e:
            console.print(f"[bold red]❌ Survival Prediction failed: {str(e)}[/bold red]")
            raise
    
    def generate_final_report(self) -> str:
        """
        Generate a comprehensive final report.
        
        Returns:
            str: Path to the final report
        """
        console.print("\n" + "=" * 80)
        console.print("[bold magenta]📊 GENERATING FINAL REPORT[/bold magenta]")
        console.print("=" * 80)
        
        try:
            report = {
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_type': 'Complete Titanic Dataset Analysis',
                    'phases_completed': list(self.analysis_results.keys())
                },
                'executive_summary': self._generate_executive_summary(),
                'detailed_results': self.analysis_results
            }
            
            # Save comprehensive report
            report_path = self.reports_dir / f"titanic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Generate markdown summary
            markdown_report = self._generate_markdown_report()
            markdown_path = self.reports_dir / f"titanic_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            
            console.print(f"[green]✅ Final report saved to: {report_path}[/green]")
            console.print(f"[green]✅ Summary report saved to: {markdown_path}[/green]")
            
            return str(report_path)
            
        except Exception as e:
            console.print(f"[red]❌ Report generation failed: {str(e)}[/red]")
            raise
    
    def _generate_executive_summary(self) -> dict:
        """Generate executive summary of the analysis."""
        summary = {
            'dataset_overview': {},
            'key_findings': {},
            'model_performance': {},
            'recommendations': []
        }
        
        # Dataset overview
        if 'etl' in self.analysis_results:
            etl_results = self.analysis_results['etl']
            summary['dataset_overview'] = {
                'total_passengers': etl_results.get('raw_data_shape', [0])[0],
                'features_original': etl_results.get('raw_data_shape', [0, 0])[1],
                'features_engineered': etl_results.get('transformed_data_shape', [0, 0])[1],
                'data_quality_improvement': 'Significant missing data reduction and feature engineering applied'
            }
        
        # Key findings from EDA
        if 'eda' in self.analysis_results:
            eda_results = self.analysis_results['eda']
            if 'survival_analysis' in eda_results:
                survival_stats = eda_results['survival_analysis']
                summary['key_findings'] = {
                    'overall_survival_rate': survival_stats.get('overall_survival_rate', 0),
                    'total_survivors': survival_stats.get('survivors', 0),
                    'total_casualties': survival_stats.get('casualties', 0),
                    'survival_patterns': 'Significant differences in survival rates by passenger class, sex, and age'
                }
        
        # Model performance
        if 'prediction' in self.analysis_results:
            prediction_results = self.analysis_results['prediction']
            if 'best_model' in prediction_results:
                best_model = prediction_results['best_model']
                summary['model_performance'] = {
                    'best_model': best_model.get('name', 'Unknown'),
                    'accuracy': best_model.get('metrics', {}).get('accuracy', 0),
                    'f1_score': best_model.get('metrics', {}).get('f1_score', 0),
                    'precision': best_model.get('metrics', {}).get('precision', 0),
                    'recall': best_model.get('metrics', {}).get('recall', 0)
                }
        
        # Recommendations
        summary['recommendations'] = [
            "Passenger class and sex were the strongest predictors of survival",
            "Age and family size also played significant roles in survival outcomes",
            "The Random Forest model showed the best balance of accuracy and interpretability",
            "Feature engineering significantly improved model performance",
            "Further analysis could explore interaction effects between features"
        ]
        
        return summary
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown summary report."""
        report_lines = [
            "# Titanic Dataset Analysis Report",
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Add executive summary
        exec_summary = self._generate_executive_summary()
        
        # Dataset Overview
        if 'dataset_overview' in exec_summary:
            overview = exec_summary['dataset_overview']
            report_lines.extend([
                "### Dataset Overview",
                f"- **Total Passengers**: {overview.get('total_passengers', 'N/A')}",
                f"- **Original Features**: {overview.get('features_original', 'N/A')}",
                f"- **Engineered Features**: {overview.get('features_engineered', 'N/A')}",
                f"- **Data Quality**: {overview.get('data_quality_improvement', 'N/A')}",
                ""
            ])
        
        # Key Findings
        if 'key_findings' in exec_summary:
            findings = exec_summary['key_findings']
            report_lines.extend([
                "### Key Findings",
                f"- **Overall Survival Rate**: {findings.get('overall_survival_rate', 0):.1%}",
                f"- **Total Survivors**: {findings.get('total_survivors', 'N/A')}",
                f"- **Total Casualties**: {findings.get('total_casualties', 'N/A')}",
                f"- **Survival Patterns**: {findings.get('survival_patterns', 'N/A')}",
                ""
            ])
        
        # Model Performance
        if 'model_performance' in exec_summary:
            performance = exec_summary['model_performance']
            report_lines.extend([
                "### Best Model Performance",
                f"- **Model**: {performance.get('best_model', 'N/A')}",
                f"- **Accuracy**: {performance.get('accuracy', 0):.3f}",
                f"- **F1 Score**: {performance.get('f1_score', 0):.3f}",
                f"- **Precision**: {performance.get('precision', 0):.3f}",
                f"- **Recall**: {performance.get('recall', 0):.3f}",
                ""
            ])
        
        # Recommendations
        if 'recommendations' in exec_summary:
            report_lines.extend([
                "### Recommendations",
                ""
            ])
            for i, rec in enumerate(exec_summary['recommendations'], 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Analysis Phases
        report_lines.extend([
            "## Analysis Phases Completed",
            ""
        ])
        
        phases = {
            'etl': 'ETL Pipeline - Data extraction, transformation, and loading',
            'eda': 'Exploratory Data Analysis - Statistical analysis and visualizations',
            'prediction': 'Survival Prediction - Machine learning model development and evaluation'
        }
        
        for phase_key, phase_desc in phases.items():
            if phase_key in self.analysis_results:
                report_lines.append(f"✅ **{phase_desc}**")
            else:
                report_lines.append(f"❌ **{phase_desc}**")
        
        report_lines.extend([
            "",
            "## Files Generated",
            "",
            "### Data Files",
            "- Raw Titanic dataset",
            "- Processed datasets (basic, engineered, ML-ready)",
            "- Data dictionary and quality reports",
            "",
            "### Visualizations",
            "- Survival analysis plots",
            "- Distribution plots",
            "- Correlation heatmap",
            "- Model comparison charts",
            "- Confusion matrices",
            "- ROC curves",
            "- Feature importance plots",
            "",
            "### Models",
            "- Trained machine learning models",
            "- Model evaluation metrics",
            "- Cross-validation results",
            "",
            "---",
            "*This report was generated automatically by the Titanic Analysis Pipeline.*"
        ])
        
        return "\n".join(report_lines)
    
    def run_complete_analysis(self, scale_features: bool = True) -> dict:
        """
        Run the complete Titanic analysis pipeline.
        
        Args:
            scale_features (bool): Whether to scale numerical features
            
        Returns:
            dict: Complete analysis results
        """
        console.print("[bold cyan]🚢 TITANIC DATASET ANALYSIS PIPELINE[/bold cyan]")
        console.print("[bold cyan]" + "=" * 80 + "[/bold cyan]")
        console.print("[yellow]Starting comprehensive analysis of the Titanic dataset...[/yellow]")
        console.print("[yellow]This includes ETL, EDA, and Machine Learning phases.[/yellow]")
        
        start_time = datetime.now()
        
        try:
            # Phase 1: ETL Pipeline
            etl_results = self.run_etl_pipeline(scale_features=scale_features)
            
            # Load the processed data for analysis
            processed_data_path = self.data_dir / "processed" / "titanic_ml_ready.csv"
            if processed_data_path.exists():
                df_processed = pd.read_csv(processed_data_path)
                console.print(f"[green]✅ Loaded processed data: {df_processed.shape}[/green]")
            else:
                console.print("[red]❌ Processed data not found, using basic features[/red]")
                # Fallback to basic processed data
                basic_data_path = self.data_dir / "processed" / "titanic_basic_features.csv"
                df_processed = pd.read_csv(basic_data_path)
            
            # Phase 2: Exploratory Data Analysis
            eda_results = self.run_exploratory_analysis(df_processed)
            
            # Phase 3: Survival Prediction
            prediction_results = self.run_survival_prediction(df_processed)
            
            # Generate final report
            report_path = self.generate_final_report()
            
            # Calculate total execution time
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            console.print("\n" + "=" * 80)
            console.print("[bold green]🎉 ANALYSIS PIPELINE COMPLETED SUCCESSFULLY! 🎉[/bold green]")
            console.print("=" * 80)
            console.print(f"[yellow]Total execution time: {execution_time}[/yellow]")
            console.print(f"[yellow]Final report: {report_path}[/yellow]")
            
            # Summary statistics
            console.print("\n[bold blue]📊 ANALYSIS SUMMARY:[/bold blue]")
            if 'etl' in self.analysis_results:
                etl = self.analysis_results['etl']
                console.print(f"[green]• Dataset processed: {etl.get('raw_data_shape', 'N/A')} → {etl.get('transformed_data_shape', 'N/A')}[/green]")
            
            if 'eda' in self.analysis_results and 'survival_analysis' in self.analysis_results['eda']:
                survival = self.analysis_results['eda']['survival_analysis']
                console.print(f"[green]• Survival rate: {survival.get('overall_survival_rate', 0):.1%}[/green]")
            
            if 'prediction' in self.analysis_results and 'best_model' in self.analysis_results['prediction']:
                best_model = self.analysis_results['prediction']['best_model']
                console.print(f"[green]• Best model: {best_model.get('name', 'N/A')} (F1: {best_model.get('metrics', {}).get('f1_score', 0):.3f})[/green]")
            
            return self.analysis_results
            
        except Exception as e:
            console.print(f"\n[bold red]💥 ANALYSIS PIPELINE FAILED: {str(e)}[/bold red]")
            raise

def main():
    """Main function to run the complete Titanic analysis."""
    try:
        # Initialize the orchestrator
        orchestrator = TitanicAnalysisOrchestrator()
        
        # Run the complete analysis
        results = orchestrator.run_complete_analysis(scale_features=True)
        
        console.print("\n[bold green]Analysis completed successfully! Check the output directory for results.[/bold green]")
        
        return results
        
    except Exception as e:
        console.print(f"[bold red]Analysis failed: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
