"""
Titanic Exploratory Data Analysis Module

This module provides comprehensive exploratory data analysis capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rich.console import Console
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
console = Console()

class TitanicEDA:
    """Class for Titanic exploratory data analysis."""
    
    def __init__(self, output_dir: str = "../../output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def basic_info(self, df: pd.DataFrame) -> dict:
        """
        Generate basic information about the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Basic information summary
        """
        console.print("[cyan]Generating basic dataset information...[/cyan]")
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        console.print(f"[green]Dataset shape: {info['shape']}[/green]")
        console.print(f"[green]Total missing values: {sum(info['missing_values'].values())}[/green]")
        console.print(f"[green]Duplicate rows: {info['duplicate_rows']}[/green]")
        console.print(f"[green]Memory usage: {info['memory_usage']:.2f} MB[/green]")
        
        return info
    
    def survival_analysis(self, df: pd.DataFrame) -> dict:
        """
        Analyze survival patterns in the dataset.
        
        Args:
            df (pd.DataFrame): Dataset with survival information
            
        Returns:
            dict: Survival analysis results
        """
        console.print("[cyan]Analyzing survival patterns...[/cyan]")
        
        if 'survived' not in df.columns:
            console.print("[red]Warning: 'survived' column not found[/red]")
            return {}
        
        survival_stats = {
            'overall_survival_rate': df['survived'].mean(),
            'total_passengers': len(df),
            'survivors': df['survived'].sum(),
            'casualties': len(df) - df['survived'].sum()
        }
        
        # Survival by categorical variables
        categorical_vars = ['sex', 'pclass', 'embarked']
        survival_by_category = {}
        
        for var in categorical_vars:
            if var in df.columns:
                survival_by_category[var] = df.groupby(var)['survived'].agg(['count', 'sum', 'mean']).round(3)
        
        survival_stats['by_category'] = survival_by_category
        
        console.print(f"[green]Overall survival rate: {survival_stats['overall_survival_rate']:.3f}[/green]")
        console.print(f"[green]Survivors: {survival_stats['survivors']} / {survival_stats['total_passengers']}[/green]")
        
        return survival_stats
    
    def numerical_analysis(self, df: pd.DataFrame) -> dict:
        """
        Analyze numerical variables in the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Numerical analysis results
        """
        console.print("[cyan]Analyzing numerical variables...[/cyan]")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'survived' in numerical_cols:
            numerical_cols.remove('survived')  # Remove target variable
        
        numerical_stats = {}
        
        for col in numerical_cols:
            stats_dict = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'q25': df[col].quantile(0.25),
                'median': df[col].median(),
                'q75': df[col].quantile(0.75),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            numerical_stats[col] = stats_dict
        
        console.print(f"[green]Analyzed {len(numerical_cols)} numerical variables[/green]")
        
        return numerical_stats
    
    def categorical_analysis(self, df: pd.DataFrame) -> dict:
        """
        Analyze categorical variables in the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Categorical analysis results
        """
        console.print("[cyan]Analyzing categorical variables...[/cyan]")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_stats = {}
        
        for col in categorical_cols:
            stats_dict = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'value_counts': df[col].value_counts().to_dict(),
                'missing_count': df[col].isnull().sum()
            }
            categorical_stats[col] = stats_dict
        
        console.print(f"[green]Analyzed {len(categorical_cols)} categorical variables[/green]")
        
        return categorical_stats
    
    def correlation_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlations between numerical variables.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        console.print("[cyan]Analyzing correlations...[/cyan]")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            console.print("[yellow]Warning: Less than 2 numerical columns for correlation analysis[/yellow]")
            return pd.DataFrame()
        
        correlation_matrix = df[numerical_cols].corr()
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        if strong_correlations:
            console.print(f"[yellow]Found {len(strong_correlations)} strong correlations (|r| > 0.7)[/yellow]")
        
        return correlation_matrix
    
    def create_survival_plots(self, df: pd.DataFrame) -> list:
        """
        Create survival analysis plots.
        
        Args:
            df (pd.DataFrame): Dataset with survival information
            
        Returns:
            list: List of saved plot filenames
        """
        console.print("[cyan]Creating survival plots...[/cyan]")
        
        if 'survived' not in df.columns:
            console.print("[red]Warning: 'survived' column not found[/red]")
            return []
        
        saved_plots = []
        
        # 1. Overall survival pie chart
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        survival_counts = df['survived'].value_counts()
        labels = ['Died', 'Survived']
        colors = ['#ff6b6b', '#4ecdc4']
        
        ax.pie(survival_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Overall Survival Rate', fontsize=16, fontweight='bold')
        
        plot_path = self.output_dir / 'survival_overall.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append(str(plot_path))
        
        # 2. Survival by passenger class
        if 'pclass' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            survival_by_class = df.groupby('pclass')['survived'].agg(['count', 'sum'])
            survival_by_class['survival_rate'] = survival_by_class['sum'] / survival_by_class['count']
            
            x = range(len(survival_by_class.index))
            ax.bar(x, survival_by_class['survival_rate'], color=['#ff6b6b', '#feca57', '#48dbfb'])
            ax.set_xlabel('Passenger Class')
            ax.set_ylabel('Survival Rate')
            ax.set_title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'Class {i}' for i in survival_by_class.index])
            
            # Add value labels on bars
            for i, v in enumerate(survival_by_class['survival_rate']):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plot_path = self.output_dir / 'survival_by_class.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(plot_path))
        
        # 3. Survival by sex
        if 'sex' in df.columns:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            survival_by_sex = df.groupby('sex')['survived'].agg(['count', 'sum'])
            survival_by_sex['survival_rate'] = survival_by_sex['sum'] / survival_by_sex['count']
            
            ax.bar(survival_by_sex.index, survival_by_sex['survival_rate'], color=['#ff6b6b', '#4ecdc4'])
            ax.set_xlabel('Sex')
            ax.set_ylabel('Survival Rate')
            ax.set_title('Survival Rate by Sex', fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for i, (idx, v) in enumerate(survival_by_sex['survival_rate'].items()):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plot_path = self.output_dir / 'survival_by_sex.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(plot_path))
        
        console.print(f"[green]Created {len(saved_plots)} survival plots[/green]")
        return saved_plots
    
    def create_distribution_plots(self, df: pd.DataFrame) -> list:
        """
        Create distribution plots for numerical variables.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            list: List of saved plot filenames
        """
        console.print("[cyan]Creating distribution plots...[/cyan]")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'survived' in numerical_cols:
            numerical_cols.remove('survived')
        
        saved_plots = []
        
        # Create subplots for distributions
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        if numerical_cols:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col.title()}', fontweight='bold')
                    axes[i].set_xlabel(col.title())
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = self.output_dir / 'distributions.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(str(plot_path))
        
        console.print(f"[green]Created distribution plots[/green]")
        return saved_plots
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> str:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            
        Returns:
            str: Path to saved plot
        """
        console.print("[cyan]Creating correlation heatmap...[/cyan]")
        
        if correlation_matrix.empty:
            console.print("[yellow]Warning: Empty correlation matrix[/yellow]")
            return ""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plot_path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print("[green]Created correlation heatmap[/green]")
        return str(plot_path)
    
    def run_complete_eda(self, df: pd.DataFrame) -> dict:
        """
        Run complete exploratory data analysis.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Complete EDA results
        """
        console.print("[bold blue]🔍 RUNNING COMPLETE EXPLORATORY DATA ANALYSIS[/bold blue]")
        console.print("=" * 60)
        
        eda_results = {}
        
        try:
            # Basic information
            eda_results['basic_info'] = self.basic_info(df)
            
            # Survival analysis
            eda_results['survival_analysis'] = self.survival_analysis(df)
            
            # Numerical analysis
            eda_results['numerical_analysis'] = self.numerical_analysis(df)
            
            # Categorical analysis
            eda_results['categorical_analysis'] = self.categorical_analysis(df)
            
            # Correlation analysis
            eda_results['correlation_matrix'] = self.correlation_analysis(df)
            
            # Create plots
            console.print("\n" + "=" * 60)
            console.print("[bold blue]Creating visualizations...[/bold blue]")
            
            survival_plots = self.create_survival_plots(df)
            distribution_plots = self.create_distribution_plots(df)
            correlation_plot = self.create_correlation_heatmap(eda_results['correlation_matrix'])
            
            eda_results['plots'] = {
                'survival_plots': survival_plots,
                'distribution_plots': distribution_plots,
                'correlation_plot': correlation_plot
            }
            
            console.print("\n" + "=" * 60)
            console.print("[bold green]🎉 EXPLORATORY DATA ANALYSIS COMPLETED![/bold green]")
            console.print(f"[yellow]Total plots created: {len(survival_plots) + len(distribution_plots) + (1 if correlation_plot else 0)}[/yellow]")
            
            return eda_results
            
        except Exception as e:
            console.print(f"[red]✗ EDA failed: {str(e)}[/red]")
            raise

def main():
    """Main function to demonstrate EDA capabilities."""
    console.print("[bold blue]Titanic Exploratory Data Analysis Module[/bold blue]")
    console.print("This module provides comprehensive exploratory data analysis capabilities.")

if __name__ == "__main__":
    main()
