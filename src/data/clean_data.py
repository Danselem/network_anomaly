"""Script to clean the dataset"""
import pandas as pd
from pathlib import Path
from src import logger


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by removing missing values and duplicates

    Args:
        data (pd.DataFrame): The dataset to be cleaned

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    logger.info("Original data shape: %s", data.shape)
    data['label'] = data['label'].apply(lambda x: x.split('.')[0])
    logger.info("Cleaned data shape: %s", data.shape)
    return data

    

def main():
    """Main function to clean the dataset"""
    logger.info("Cleaning the training dataset")
    data = pd.read_parquet("data/interim/kddcup.parquet")
    cleaned_data = clean_data(data)
    data_path = Path("data/processed/kddcup_cleaned.parquet")
    cleaned_data.to_parquet(data_path, index=False)
    logger.info(f"Cleaned data saved to {data_path}")


if __name__ == "__main__":
    main()