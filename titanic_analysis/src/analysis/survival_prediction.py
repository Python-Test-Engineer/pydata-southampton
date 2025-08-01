"""
Titanic Survival Prediction Module

This module provides machine learning models for predicting passenger survival.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings('ignore')
console = Console()

class TitanicSurvivalPredictor:
    """Class for Titanic survival prediction using machine learning."""
    
    def __init__(self, output_dir: str = "../../output"):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.models_dir = self.output_dir / "models"
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True)
        }
        
        self.trained_models = {}
        self.model_results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for machine learning.
        
        Args:
            df (pd.DataFrame): Dataset with features
            
        Returns:
            tuple: (X, y) features and target
        """
        console.print("[cyan]Preparing features for machine learning...[/cyan]")
        
        if 'survived' not in df.columns:
            raise ValueError("Target variable 'survived' not found in dataset")
        
        # Separate features and target
        X = df.drop('survived', axis=1)
        y = df['survived']
        
        # Select only numerical features for ML models
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # Handle any remaining missing values
        X_numerical = X_numerical.fillna(X_numerical.median())
        
        console.print(f"[green]Features prepared: {X_numerical.shape[1]} numerical features[/green]")
        console.print(f"[green]Target distribution: {y.value_counts().to_dict()}[/green]")
        
        return X_numerical, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        console.print(f"[cyan]Splitting data (test_size={test_size})...[/cyan]")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        console.print(f"[green]Training set: {X_train.shape[0]} samples[/green]")
        console.print(f"[green]Test set: {X_test.shape[0]} samples[/green]")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train all machine learning models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            dict: Trained models
        """
        console.print("[cyan]Training machine learning models...[/cyan]")
        
        for name, model in self.models.items():
            console.print(f"Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                console.print(f"[green]✓ {name} trained successfully[/green]")
                
                # Save model
                model_path = self.models_dir / f"{name}_model.joblib"
                joblib.dump(model, model_path)
                
            except Exception as e:
                console.print(f"[red]✗ Error training {name}: {str(e)}[/red]")
        
        console.print(f"[green]Successfully trained {len(self.trained_models)} models[/green]")
        return self.trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate all trained models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Model evaluation results
        """
        console.print("[cyan]Evaluating models...[/cyan]")
        
        results = {}
        
        for name, model in self.trained_models.items():
            console.print(f"Evaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                results[name] = {
                    'metrics': metrics,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                console.print(f"[green]✓ {name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error evaluating {name}: {str(e)}[/red]")
        
        self.model_results = results
        return results
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """
        Perform cross-validation on all models.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        console.print(f"[cyan]Performing {cv}-fold cross-validation...[/cyan]")
        
        cv_results = {}
        
        for name, model in self.models.items():
            console.print(f"Cross-validating {name}...")
            
            try:
                # Perform cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                
                cv_results[name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_cv_score': cv_scores.mean(),
                    'std_cv_score': cv_scores.std()
                }
                
                console.print(f"[green]✓ {name} - CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})[/green]")
                
            except Exception as e:
                console.print(f"[red]✗ Error in cross-validation for {name}: {str(e)}[/red]")
        
        return cv_results
    
    def get_feature_importance(self, X: pd.DataFrame) -> dict:
        """
        Get feature importance from tree-based models.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            dict: Feature importance for each model
        """
        console.print("[cyan]Extracting feature importance...[/cyan]")
        
        feature_importance = {}
        
        for name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                feature_importance[name] = importance_df.to_dict('records')
                console.print(f"[green]✓ Feature importance extracted for {name}[/green]")
            
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                feature_importance[name] = importance_df.to_dict('records')
                console.print(f"[green]✓ Feature importance extracted for {name}[/green]")
        
        return feature_importance
    
    def create_model_comparison_plot(self) -> str:
        """
        Create model comparison visualization.
        
        Returns:
            str: Path to saved plot
        """
        console.print("[cyan]Creating model comparison plot...[/cyan]")
        
        if not self.model_results:
            console.print("[yellow]Warning: No model results available[/yellow]")
            return ""
        
        # Extract metrics for comparison
        models = list(self.model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.model_results[model]['metrics'][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
            axes[i].set_title(f'Model Comparison - {metric.title()}', fontweight='bold')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = self.figures_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print("[green]Created model comparison plot[/green]")
        return str(plot_path)
    
    def create_confusion_matrices(self) -> str:
        """
        Create confusion matrices for all models.
        
        Returns:
            str: Path to saved plot
        """
        console.print("[cyan]Creating confusion matrices...[/cyan]")
        
        if not self.model_results:
            console.print("[yellow]Warning: No model results available[/yellow]")
            return ""
        
        n_models = len(self.model_results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, results) in enumerate(self.model_results.items()):
            if i < len(axes):
                cm = np.array(results['metrics']['confusion_matrix'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                           xticklabels=['Died', 'Survived'],
                           yticklabels=['Died', 'Survived'])
                
                axes[i].set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.figures_dir / 'confusion_matrices.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print("[green]Created confusion matrices[/green]")
        return str(plot_path)
    
    def create_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """
        Create ROC curves for all models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            str: Path to saved plot
        """
        console.print("[cyan]Creating ROC curves...[/cyan]")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        for i, (name, model) in enumerate(self.trained_models.items()):
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                ax.plot(fpr, tpr, color=colors[i % len(colors)], 
                       label=f'{name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = self.figures_dir / 'roc_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print("[green]Created ROC curves[/green]")
        return str(plot_path)
    
    def create_feature_importance_plot(self, feature_importance: dict) -> str:
        """
        Create feature importance visualization.
        
        Args:
            feature_importance (dict): Feature importance data
            
        Returns:
            str: Path to saved plot
        """
        console.print("[cyan]Creating feature importance plot...[/cyan]")
        
        if not feature_importance:
            console.print("[yellow]Warning: No feature importance data available[/yellow]")
            return ""
        
        # Use Random Forest feature importance (most interpretable)
        if 'random_forest' in feature_importance:
            importance_data = feature_importance['random_forest']
            
            # Get top 10 features
            top_features = importance_data[:10]
            
            features = [item['feature'] for item in top_features]
            importances = [item['importance'] for item in top_features]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            bars = ax.barh(features, importances, color='skyblue')
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Feature Importance (Random Forest)', fontweight='bold')
            ax.invert_yaxis()
            
            # Add value labels
            for bar, importance in zip(bars, importances):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance:.3f}', ha='left', va='center')
            
            plot_path = self.figures_dir / 'feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            console.print("[green]Created feature importance plot[/green]")
            return str(plot_path)
        
        return ""
    
    def run_complete_prediction_analysis(self, df: pd.DataFrame) -> dict:
        """
        Run complete survival prediction analysis.
        
        Args:
            df (pd.DataFrame): Dataset for prediction
            
        Returns:
            dict: Complete prediction analysis results
        """
        console.print("[bold blue]🤖 RUNNING COMPLETE SURVIVAL PREDICTION ANALYSIS[/bold blue]")
        console.print("=" * 60)
        
        analysis_results = {}
        
        try:
            # Prepare features
            X, y = self.prepare_features(df)
            analysis_results['feature_shape'] = X.shape
            analysis_results['target_distribution'] = y.value_counts().to_dict()
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            # Train models
            trained_models = self.train_models(X_train, y_train)
            analysis_results['trained_models'] = list(trained_models.keys())
            
            # Evaluate models
            model_results = self.evaluate_models(X_test, y_test)
            analysis_results['model_results'] = model_results
            
            # Cross-validation
            cv_results = self.cross_validate_models(X, y)
            analysis_results['cv_results'] = cv_results
            
            # Feature importance
            feature_importance = self.get_feature_importance(X)
            analysis_results['feature_importance'] = feature_importance
            
            # Create visualizations
            console.print("\n" + "=" * 60)
            console.print("[bold blue]Creating visualizations...[/bold blue]")
            
            comparison_plot = self.create_model_comparison_plot()
            confusion_plot = self.create_confusion_matrices()
            roc_plot = self.create_roc_curves(X_test, y_test)
            importance_plot = self.create_feature_importance_plot(feature_importance)
            
            analysis_results['plots'] = {
                'model_comparison': comparison_plot,
                'confusion_matrices': confusion_plot,
                'roc_curves': roc_plot,
                'feature_importance': importance_plot
            }
            
            # Find best model
            best_model_name = max(model_results.keys(), 
                                key=lambda x: model_results[x]['metrics']['f1_score'])
            analysis_results['best_model'] = {
                'name': best_model_name,
                'metrics': model_results[best_model_name]['metrics']
            }
            
            console.print("\n" + "=" * 60)
            console.print("[bold green]🎉 SURVIVAL PREDICTION ANALYSIS COMPLETED![/bold green]")
            console.print(f"[yellow]Best model: {best_model_name}[/yellow]")
            console.print(f"[yellow]Best F1 score: {analysis_results['best_model']['metrics']['f1_score']:.3f}[/yellow]")
            
            return analysis_results
            
        except Exception as e:
            console.print(f"[red]✗ Prediction analysis failed: {str(e)}[/red]")
            raise

def main():
    """Main function to demonstrate prediction capabilities."""
    console.print("[bold blue]Titanic Survival Prediction Module[/bold blue]")
    console.print("This module provides machine learning capabilities for survival prediction.")

if __name__ == "__main__":
    main()
