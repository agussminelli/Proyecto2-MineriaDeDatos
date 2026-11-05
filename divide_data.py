import pandas as pd
from sklearn.model_selection import train_test_split
from relable import relabel_categories 

def create_stratified_partitions(raw_data, headers, target_col="gs_text34"):
    """
    Converts raw list data to a DataFrame, relabels target classes to broader categories,
    filters out rare classes (< 7 instances), and splits the dataset into 
    70% Train, 20% Dev, and 10% Test using stratified sampling.
    """
    
    # 1. Convert to Pandas DataFrame to allow stratification
    df = pd.DataFrame(raw_data, columns=headers)
    print(f"Original dataset rows: {len(df)}")
    
    # 2. Relabel the data to broader categories
    df = relabel_categories(df, target_col=target_col)
    
    # 3. First split: 70% Train and 30% Temporary (Dev + Test)
    # Using random state so the whole team gets the exact same data split
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df[target_col], 
        random_state=42 
    )
    
    # 4. Second split: Divide the 30% Temporary into 20% Dev and 10% Test
    dev_df, test_df = train_test_split(
        temp_df, 
        test_size=1/3, 
        stratify=temp_df[target_col], 
        random_state=42
    )
    
    print(f"\nPartition sizes:")
    print(f"Train: {len(train_df)} ({len(train_df)/len(df):.0%})")
    print(f"Dev:   {len(dev_df)} ({len(dev_df)/len(df):.0%})")
    print(f"Test:  {len(test_df)} ({len(test_df)/len(df):.0%})")
    
    return train_df, dev_df, test_df