import pandas as pd
import requests
from pathlib import Path
from src import logger  # Assuming logger is set up and imported from src

def download_file(url: str, save_dir: str, file_name: str):
    """
    Download a file from a URL and save it to a specified directory.

    Parameters:
    url (str): The URL of the file to download.
    save_dir (str): The directory to save the file in.
    file_name (str): The name of the file to save the data as.
    """
    # Create the save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Define the full path to save the file
    local_path = save_path / file_name

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"File successfully downloaded from {url} to {local_path}")
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {e}")
        raise

def ingest_data(file_path: str, save_dir: str = 'data/interim'):
    """
    Ingest data from a gzip-compressed CSV file and save it to a 
    specified directory as a Parquet file.

    Parameters:
    file_path (str): The path of the file to read the data from.
    save_dir (str): The directory to save the data in. Defaults to 'data/raw'.
    """
    
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
        "num_failed_logins","logged_in", "num_compromised", "root_shell", 
        "su_attempted", "num_root", "num_file_creations", "num_shells", 
        "num_access_files", "num_outbound_cmds", "is_host_login", 
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate", "diff_srv_rate", 
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    try:
        # Use compression='gzip' to read the .gz file
        df = pd.read_csv(file_path, 
                        header=None, names=col_names, 
                        index_col=False,)
        logger.info(f"Data successfully read from {file_path}")
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        return

    # Create the save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Define the full path to save the file, with the .parquet extension
    parquet_path = save_path / (Path(file_path).stem + '.parquet')

    # Save the DataFrame to the specified file path in Parquet format
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Data successfully saved to {parquet_path}")
    except Exception as e:
        logger.error(f"Error saving data to {parquet_path}: {e}")

def main():
    url = 'https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz' 
    raw_dir = 'data/raw' 
    interim_dir = 'data/interim'
    file_name = 'kddcup.gz'
    
    try:
        download_file(url, save_dir=raw_dir, file_name=file_name)
        local_file_path = Path(raw_dir) / file_name
        ingest_data(local_file_path, save_dir=interim_dir)
    except Exception as e:
        logger.error(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    main()