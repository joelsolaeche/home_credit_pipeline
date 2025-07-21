from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"] = working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_val_df["DAYS_EMPLOYED"] = working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan})
    working_test_df["DAYS_EMPLOYED"] = working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # 2. Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    
    # Get categorical columns (object type)
    categorical_columns = working_train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Separate binary and multi-category columns
    binary_columns = []
    multi_category_columns = []
    
    for col in categorical_columns:
        if working_train_df[col].nunique() == 2:
            binary_columns.append(col)
        else:
            multi_category_columns.append(col)
    
    # Binary encoding using OrdinalEncoder
    if binary_columns:
        binary_encoder = OrdinalEncoder()
        binary_encoder.fit(working_train_df[binary_columns])
        
        # Transform all datasets
        working_train_df[binary_columns] = binary_encoder.transform(working_train_df[binary_columns])
        working_val_df[binary_columns] = binary_encoder.transform(working_val_df[binary_columns])
        working_test_df[binary_columns] = binary_encoder.transform(working_test_df[binary_columns])
    
    # Multi-category encoding using OneHotEncoder
    if multi_category_columns:
        # Use sparse_output instead of sparse for newer sklearn versions
        try:
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # Fallback for older sklearn versions
            onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        onehot_encoder.fit(working_train_df[multi_category_columns])
        
        # Transform and get column names
        train_encoded = onehot_encoder.transform(working_train_df[multi_category_columns])
        val_encoded = onehot_encoder.transform(working_val_df[multi_category_columns])
        test_encoded = onehot_encoder.transform(working_test_df[multi_category_columns])
        
        # Get feature names
        feature_names = onehot_encoder.get_feature_names_out(multi_category_columns)
        
        # Create DataFrames with the encoded values
        train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=working_train_df.index)
        val_encoded_df = pd.DataFrame(val_encoded, columns=feature_names, index=working_val_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=working_test_df.index)
        
        # Drop original categorical columns and add encoded ones
        working_train_df = working_train_df.drop(columns=multi_category_columns).join(train_encoded_df)
        working_val_df = working_val_df.drop(columns=multi_category_columns).join(val_encoded_df)
        working_test_df = working_test_df.drop(columns=multi_category_columns).join(test_encoded_df)

    # 3. Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value.
    imputer = SimpleImputer(strategy='median')
    imputer.fit(working_train_df)
    
    # Transform all datasets
    working_train_df_imputed = pd.DataFrame(
        imputer.transform(working_train_df),
        columns=working_train_df.columns,
        index=working_train_df.index
    )
    working_val_df_imputed = pd.DataFrame(
        imputer.transform(working_val_df),
        columns=working_val_df.columns,
        index=working_val_df.index
    )
    working_test_df_imputed = pd.DataFrame(
        imputer.transform(working_test_df),
        columns=working_test_df.columns,
        index=working_test_df.index
    )

    # 4. Feature scaling with Min-Max scaler. Apply this to all the columns.
    scaler = MinMaxScaler()
    scaler.fit(working_train_df_imputed)
    
    # Transform all datasets
    train_scaled = scaler.transform(working_train_df_imputed)
    val_scaled = scaler.transform(working_val_df_imputed)
    test_scaled = scaler.transform(working_test_df_imputed)
    
    print("Output train data shape: ", train_scaled.shape)
    print("Output val data shape: ", val_scaled.shape)
    print("Output test data shape: ", test_scaled.shape)

    return train_scaled, val_scaled, test_scaled
