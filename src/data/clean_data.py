"""Script to clean the dataset"""
import pandas as pd
from pathlib import Path
from src import logger


def create_directory_if_not_exists(directory: str):
    """Creates a directory if it does not exist"""
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory '{directory}' created.")
    else:
        logger.info(f"Directory '{directory}' already exists.")


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
    # Check and create interim and processed directories if necessary
    create_directory_if_not_exists("data/interim")
    create_directory_if_not_exists("data/processed")

    logger.info("Cleaning the training dataset")
    
    # Load the data
    data = pd.read_parquet("data/interim/kddcup.parquet")
    
    # Clean the data
    cleaned_data = clean_data(data)
    
    # Save the cleaned data
    data_path = Path("data/processed/kddcup_cleaned.parquet")
    cleaned_data.to_parquet(data_path, index=False)
    
    logger.info(f"Cleaned data saved to {data_path}")


if __name__ == "__main__":
    main()