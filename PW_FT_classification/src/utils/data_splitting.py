## DATA SPLITTING

import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def create_splits(csv_path, output_folder, test_size=0.2, val_size=0.1):
    """
    Create stratified training, validation, and testing splits.
    
    Args:
    - csv_path (str): Path to the csv containing the annotations.
    - output_folder (str): Destination directory to save the annotation split csv files.
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the training dataset to include in the validation split.
    
    Returns:
    - A tuple of DataFrames: (train_set, val_set, test_set)
    - Saves the splits into separate csv files in the output_folder.
    """
    # Load the data from the csv file
    data = pd.read_csv(csv_path)
    # Separate the features and the targets
    X = data[['path','label']]
    y = data['classification']
    
    # First split to separate out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    # Adjust val_size to account for the initial split
    val_size_adjusted = val_size / (1 - test_size)
    
    # Second split to separate out the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42)
    
    # Combine features, labels, and classification back into dataframes
    train_set = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_set = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_set = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    
    # Create the output directory in case that it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the splits to new CSV files
    train_set.to_csv(os.path.join(output_folder,'train_annotations.csv'), index=False)
    val_set.to_csv(os.path.join(output_folder,'val_annotations.csv'), index=False)
    test_set.to_csv(os.path.join(output_folder,'test_annotations.csv'), index=False)

    # Return the dataframes
    return train_set, val_set, test_set

def split_by_location(csv_path, output_folder, val_size=0.15, test_size=0.15, random_state=None):
    """
    Splits the dataset into train, validation, and test sets based on location, ensuring that:
    1. All images from the same location are in the same split.
    2. The split is random among the locations.
    3. Saves the split datasets into CSV files.
    
    Parameters:
    - csv_path: Path to the csv containing the annotations.
    - train_size, val_size, test_size: float, proportions of the dataset to include in the train, validation, and test splits.
    - random_state: int, random state for reproducibility.
    """
    # Load the data from the csv file
    data = pd.read_csv(csv_path)

    # Calculate train size based on val and test size
    train_size = 1.0 - val_size - test_size
    
    # Get unique locations
    unique_locations = data['Location'].unique()

    # Split locations into train and temp (temporary holding for val and test)
    train_locs, temp_locs = train_test_split(unique_locations, train_size=train_size, random_state=random_state)
    
    # Adjust the proportions for val and test based on the remaining locations
    temp_size = val_size / (val_size + test_size)
    val_locs, test_locs = train_test_split(temp_locs, train_size=temp_size, random_state=random_state)
    
    # Allocate images to train, validation, and test sets based on their location
    train_data = data[data['Location'].isin(train_locs)]
    val_data = data[data['Location'].isin(val_locs)]
    test_data = data[data['Location'].isin(test_locs)]
    
    # Save the datasets to CSV files
    train_data.to_csv(os.path.join(output_folder,'train_annotations.csv'), index=False)
    val_data.to_csv(os.path.join(output_folder,'val_annotations.csv'), index=False)
    test_data.to_csv(os.path.join(output_folder,'test_annotations.csv'), index=False)
    
    # Return the split datasets
    return train_data, val_data, test_data


def split_by_seq(csv_path, output_folder, val_size=0.15, test_size=0.15, random_state=None):
    """
    Splits the dataset into train, validation, and test sets based on sequence ID, ensuring that:
    1. All images from the same sequence are in the same split.
    2. The split is random among the sequences.
    3. Saves the split datasets into CSV files.
    
    Parameters:
    - csv_path: Path to the csv containing the annotations.
    - train_size, val_size, test_size: float, proportions of the dataset to include in the train, validation, and test splits.
    - random_state: int, random state for reproducibility.
    """
    # Load the data from the csv file
    data = pd.read_csv(csv_path)

    # Convert 'Photo_Time' from string to datetime
    data['Photo_Time'] = pd.to_datetime(data['Photo_Time'])

    # Calculate train size based on val and test size
    train_size = 1 - val_size - test_size
    
    # Sort by 'Photo_Time' to ensure chronological order
    data = data.sort_values(by=['Photo_Time']).reset_index(drop=True)

    # Group photos into sequences based on a 30-second interval
    time_groups = data.groupby(pd.Grouper(key='Photo_Time', freq='30S'))

    # Assign unique sequence IDs to each group
    for s, i in tqdm(enumerate(time_groups.indices.values())):
        data.loc[i, 'Seq_ID'] = int(s)

    # Get unique sequence IDs
    unique_seq_ids = data['Seq_ID'].unique()
    
    # Split sequence IDs into train and temp (temporary holding for val and test)
    train_seq_ids, temp_seq_ids = train_test_split(unique_seq_ids, train_size=train_size, random_state=random_state)
    
    # Adjust the proportions for val and test based on the remaining sequences
    temp_size = val_size / (val_size + test_size)
    val_seq_ids, test_seq_ids = train_test_split(temp_seq_ids, train_size=temp_size, random_state=random_state)
    
    # Allocate images to train, validation, and test sets based on their sequence ID
    train_data = data[data['Seq_ID'].isin(train_seq_ids)]
    val_data = data[data['Seq_ID'].isin(val_seq_ids)]
    test_data = data[data['Seq_ID'].isin(test_seq_ids)]
    
    # Save the datasets to CSV files
    train_data.to_csv(os.path.join(output_folder,'train_annotations.csv'), index=False)
    val_data.to_csv(os.path.join(output_folder,'val_annotations.csv'), index=False)
    test_data.to_csv(os.path.join(output_folder,'test_annotations.csv'), index=False)

    # Return the split datasets
    return train_data, val_data, test_data
