"""
This was developed by uisng `cline` and allowing it to create the app.

It gets Claude 3.7 to clean a CSV dataset using custom instructions. This will be expensive in the real world, but is a demo of vibe coding as well as using the Anthropic API.

Usage:
    python 30_cline_csv_cleaner_agent.py --input 30.1_dirty_sales_data_10.csv --output 30.3_cleaned_sales_data_10.csv

Requirements:
    - anthropic
    - pandas
    - python-dotenv
"""

import os
import argparse
import json
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
import time
import sys

# Load environment variables from .env file
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class CSVCleanerAgent:
    """Agent that uses Claude 3.7 to clean CSV datasets."""

    def __init__(self, api_key=None, model="claude-3-7-sonnet-20240620"):
        """
        Initialize the CSV Cleaner Agent.

        Args:
            api_key (str, optional): Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
            model (str, optional): Claude model to use. Defaults to claude-3-7-sonnet-20240620.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it directly."
            )

        self.model = model
        self.client = Anthropic(api_key=self.api_key)

    def load_csv(self, file_path):
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)

    def save_csv(self, df, file_path):
        """
        Save a pandas DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): The DataFrame to save.
            file_path (str): Path where the CSV file will be saved.
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Cleaned data saved to {file_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            sys.exit(1)

    def get_data_summary(self, df):
        """
        Generate a summary of the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to summarize.

        Returns:
            str: A summary of the DataFrame.
        """
        summary = []
        summary.append(f"Shape: {df.shape}")
        summary.append("\nColumn names and data types:")
        summary.append(str(df.dtypes))
        summary.append("\nSample data (first 5 rows):")
        summary.append(str(df.head()))
        summary.append("\nMissing values per column:")
        summary.append(str(df.isnull().sum()))
        summary.append("\nBasic statistics:")
        summary.append(str(df.describe(include="all").to_string()))

        return "\n".join(summary)

    def clean_data_with_claude(self, df, custom_instructions=None):
        """
        Use Claude 3.7 to clean the data.

        Args:
            df (pandas.DataFrame): The DataFrame to clean.
            custom_instructions (str, optional): Custom cleaning instructions.

        Returns:
            pandas.DataFrame: The cleaned DataFrame.
        """
        # Generate data summary
        data_summary = self.get_data_summary(df)

        # Convert a sample of the DataFrame to CSV string for Claude
        sample_csv = df.head(20).to_csv(index=False)

        # Default cleaning instructions
        default_instructions = """
        You are a data cleaning expert. Your task is to clean the provided CSV data and return the cleaned version.
        
        Please perform the following cleaning operations:
        1. Fix any misspelled column names
        2. Standardize date formats (use ISO format YYYY-MM-DD for dates)
        3. Handle missing values appropriately (impute or mark as needed)
        4. Fix any inconsistent or incorrect data
        5. Standardize text case where appropriate (e.g., proper case for names)
        6. Remove any duplicate records
        7. Ensure numeric columns have the correct data type
        8. Ensure categorical columns are properly formatted
        9. Rename columns to follow a consistent naming convention if needed
        
        Return your response in the following JSON format:
        {
            "cleaning_operations": [
                {"operation": "description of what was done", "details": "specific details about the change"}
            ],
            "cleaned_data_csv": "the entire cleaned dataset as a CSV string"
        }
        """

        # Combine default and custom instructions
        instructions = default_instructions
        if custom_instructions:
            instructions += (
                f"\n\nAdditional custom instructions:\n{custom_instructions}"
            )

        # Prepare the message for Claude
        prompt = f"""
        {instructions}
        
        Here is the data summary:
        {data_summary}
        
        Here is a sample of the CSV data:
        {sample_csv}
        
        Now, please clean the entire dataset and return the cleaned version in the specified JSON format.
        """

        print("Sending data to Claude 3.7 for cleaning...")
        start_time = time.time()

        try:
            # Send the request to Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                system="You are a data cleaning expert that helps clean CSV datasets. You always return your response in the exact JSON format requested.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the response content
            response_content = response.content[0].text

            # Parse the JSON response
            try:
                # Try to find JSON in the response
                json_start = response_content.find("{")
                json_end = response_content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in the response")

                # Extract the cleaned data CSV string
                cleaned_data_csv = result.get("cleaned_data_csv")
                if not cleaned_data_csv:
                    raise ValueError("No cleaned data found in the response")

                # Convert the CSV string back to a DataFrame
                import io

                cleaned_df = pd.read_csv(io.StringIO(cleaned_data_csv))

                # Print the cleaning operations
                print("\nCleaning operations performed:")
                for op in result.get("cleaning_operations", []):
                    print(f"- {op.get('operation')}: {op.get('details')}")

                elapsed_time = time.time() - start_time
                print(f"\nData cleaning completed in {elapsed_time:.2f} seconds")

                return cleaned_df

            except Exception as e:
                print(f"Error parsing Claude's response: {e}")
                print("Raw response:", response_content)
                sys.exit(1)

        except Exception as e:
            print(f"Error communicating with Claude API: {e}")
            sys.exit(1)

    def clean_csv(self, input_file, output_file, custom_instructions=None):
        """
        Clean a CSV file using Claude 3.7.

        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path where the cleaned CSV file will be saved.
            custom_instructions (str, optional): Custom cleaning instructions.
        """
        print(f"Loading CSV file: {input_file}")
        df = self.load_csv(input_file)

        print(f"CSV loaded successfully. Shape: {df.shape}")

        cleaned_df = self.clean_data_with_claude(df, custom_instructions)

        self.save_csv(cleaned_df, output_file)

        # Print before and after comparison
        print("\nComparison of original and cleaned data:")
        print(f"Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
        print(f"Original columns: {list(df.columns)}")
        print(f"Cleaned columns: {list(cleaned_df.columns)}")

        return cleaned_df


def main():
    """Main function to run the CSV Cleaner Agent."""
    parser = argparse.ArgumentParser(description="Clean a CSV file using Claude 3.7")
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--output", required=True, help="Path where the cleaned CSV file will be saved"
    )
    parser.add_argument("--instructions", help="Custom cleaning instructions")
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (optional, can use ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model", default="claude-3-7-sonnet-20240620", help="Claude model to use"
    )

    args = parser.parse_args()

    try:
        agent = CSVCleanerAgent(
            api_key=ANTHROPIC_API_KEY, model="claude-3-7-sonnet-20250219"
        )
        agent.clean_csv(args.input, args.output, args.instructions)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
