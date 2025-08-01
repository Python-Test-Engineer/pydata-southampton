"""
Titanic Dataset Transformation Module

This module handles data cleaning, feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from rich.console import Console
import re

console = Console()

class TitanicTransformer:
    """Class to handle all Titanic dataset transformations."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the Titanic dataset by handling missing values and data quality issues.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        console.print("[cyan]Starting data cleaning...[/cyan]")
        df_clean = df.copy()
        
        # Handle missing values
        console.print("Handling missing values...")
        
        # Age: Fill with median age by passenger class and sex
        age_median = df_clean.groupby(['pclass', 'sex'])['age'].median()
        for (pclass, sex), median_age in age_median.items():
            mask = (df_clean['pclass'] == pclass) & (df_clean['sex'] == sex) & df_clean['age'].isna()
            df_clean.loc[mask, 'age'] = median_age
        
        # Embarked: Fill with mode (most common port)
        embarked_mode = df_clean['embarked'].mode()[0]
        df_clean['embarked'].fillna(embarked_mode, inplace=True)
        
        # Fare: Fill with median fare by passenger class
        fare_median = df_clean.groupby('pclass')['fare'].median()
        for pclass, median_fare in fare_median.items():
            mask = (df_clean['pclass'] == pclass) & df_clean['fare'].isna()
            df_clean.loc[mask, 'fare'] = median_fare
        
        # Deck: Extract from cabin and handle missing values
        if 'cabin' in df_clean.columns:
            df_clean['deck'] = df_clean['cabin'].str[0]
            df_clean['deck'].fillna('Unknown', inplace=True)
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        console.print(f"[green]✓ Data cleaning completed. Removed {removed_duplicates} duplicate rows[/green]")
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            df (pd.DataFrame): Cleaned dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        console.print("[cyan]Starting feature engineering...[/cyan]")
        df_features = df.copy()
        
        # Family size
        df_features['family_size'] = df_features['sibsp'] + df_features['parch'] + 1
        
        # Is alone
        df_features['is_alone'] = (df_features['family_size'] == 1).astype(int)
        
        # Age groups
        df_features['age_group'] = pd.cut(df_features['age'], 
                                        bins=[0, 12, 18, 35, 60, 100], 
                                        labels=['Child', 'Teen', 'Adult', 'Middle_Age', 'Senior'])
        
        # Fare per person
        df_features['fare_per_person'] = df_features['fare'] / df_features['family_size']
        
        # Title extraction from name
        if 'who' in df_features.columns:
            # Seaborn dataset already has 'who' column (man, woman, child)
            df_features['title'] = df_features['who']
        else:
            # Extract title from name if available
            if 'name' in df_features.columns:
                df_features['title'] = df_features['name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
                # Group rare titles
                title_mapping = {
                    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                    'Capt': 'Rare', 'Sir': 'Rare'
                }
                df_features['title'] = df_features['title'].map(title_mapping).fillna('Rare')
        
        # Cabin availability
        if 'cabin' in df_features.columns:
            df_features['has_cabin'] = df_features['cabin'].notna().astype(int)
        
        # Fare bins
        df_features['fare_bin'] = pd.qcut(df_features['fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
        
        console.print("[green]✓ Feature engineering completed[/green]")
        
        return df_features
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.
        
        Args:
            df (pd.DataFrame): Dataset with engineered features
            
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        console.print("[cyan]Encoding categorical features...[/cyan]")
        df_encoded = df.copy()
        
        # Binary encoding for sex
        df_encoded['sex_male'] = (df_encoded['sex'] == 'male').astype(int)
        
        # One-hot encoding for embarked
        embarked_dummies = pd.get_dummies(df_encoded['embarked'], prefix='embarked')
        df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
        
        # One-hot encoding for passenger class
        pclass_dummies = pd.get_dummies(df_encoded['pclass'], prefix='pclass')
        df_encoded = pd.concat([df_encoded, pclass_dummies], axis=1)
        
        # One-hot encoding for age group
        if 'age_group' in df_encoded.columns:
            age_group_dummies = pd.get_dummies(df_encoded['age_group'], prefix='age_group')
            df_encoded = pd.concat([df_encoded, age_group_dummies], axis=1)
        
        # One-hot encoding for title
        if 'title' in df_encoded.columns:
            title_dummies = pd.get_dummies(df_encoded['title'], prefix='title')
            df_encoded = pd.concat([df_encoded, title_dummies], axis=1)
        
        # One-hot encoding for fare bin
        if 'fare_bin' in df_encoded.columns:
            fare_bin_dummies = pd.get_dummies(df_encoded['fare_bin'], prefix='fare_bin')
            df_encoded = pd.concat([df_encoded, fare_bin_dummies], axis=1)
        
        # One-hot encoding for deck
        if 'deck' in df_encoded.columns:
            deck_dummies = pd.get_dummies(df_encoded['deck'], prefix='deck')
            df_encoded = pd.concat([df_encoded, deck_dummies], axis=1)
        
        console.print("[green]✓ Categorical encoding completed[/green]")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, features_to_scale: list = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Dataset with encoded features
            features_to_scale (list): List of features to scale
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        console.print("[cyan]Scaling numerical features...[/cyan]")
        df_scaled = df.copy()
        
        if features_to_scale is None:
            features_to_scale = ['age', 'fare', 'fare_per_person', 'family_size']
        
        # Only scale features that exist in the dataset
        features_to_scale = [f for f in features_to_scale if f in df_scaled.columns]
        
        if features_to_scale:
            df_scaled[features_to_scale] = self.scaler.fit_transform(df_scaled[features_to_scale])
            console.print(f"[green]✓ Scaled features: {features_to_scale}[/green]")
        
        return df_scaled
    
    def transform(self, df: pd.DataFrame, scale_features: bool = True) -> pd.DataFrame:
        """
        Apply all transformations to the dataset.
        
        Args:
            df (pd.DataFrame): Raw dataset
            scale_features (bool): Whether to scale numerical features
            
        Returns:
            pd.DataFrame: Fully transformed dataset
        """
        console.print("[bold blue]Starting complete transformation pipeline...[/bold blue]")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Engineer features
        df_features = self.engineer_features(df_clean)
        
        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)
        
        # Step 4: Scale numerical features (optional)
        if scale_features:
            df_final = self.scale_numerical_features(df_encoded)
        else:
            df_final = df_encoded
        
        console.print(f"[bold green]✓ Transformation completed! Dataset shape: {df_final.shape}[/bold green]")
        
        return df_final
    
    def get_feature_summary(self, df_original: pd.DataFrame, df_transformed: pd.DataFrame) -> dict:
        """
        Generate a summary of the transformation process.
        
        Args:
            df_original (pd.DataFrame): Original dataset
            df_transformed (pd.DataFrame): Transformed dataset
            
        Returns:
            dict: Transformation summary
        """
        summary = {
            'original_shape': df_original.shape,
            'transformed_shape': df_transformed.shape,
            'original_columns': list(df_original.columns),
            'transformed_columns': list(df_transformed.columns),
            'new_features': [col for col in df_transformed.columns if col not in df_original.columns],
            'missing_values_original': df_original.isnull().sum().sum(),
            'missing_values_transformed': df_transformed.isnull().sum().sum()
        }
        
        console.print("\n[bold yellow]Transformation Summary:[/bold yellow]")
        console.print(f"Original shape: {summary['original_shape']}")
        console.print(f"Transformed shape: {summary['transformed_shape']}")
        console.print(f"New features created: {len(summary['new_features'])}")
        console.print(f"Missing values reduced from {summary['missing_values_original']} to {summary['missing_values_transformed']}")
        
        return summary

def main():
    """Main function to demonstrate transformation."""
    # This would typically be called from the main ETL pipeline
    console.print("[bold blue]Titanic Data Transformation Module[/bold blue]")
    console.print("This module provides data cleaning and feature engineering capabilities.")

if __name__ == "__main__":
    main()
