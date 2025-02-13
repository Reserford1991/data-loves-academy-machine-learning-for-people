import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

def drop_unnecessary_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops unnecessary columns from a DataFrame if they exist.

    :param df: The DataFrame to process.
    :param columns: List of column names to drop.
    :return: DataFrame with specified columns removed.
    """
    return df.drop(columns=[col for col in columns if col in df], axis=1)

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and validation sets.

    :param df: The DataFrame to split.
    :param test_size: Proportion of data to be used as validation set.
    :param random_state: Seed for reproducibility.
    :return: A tuple containing train_df and val_df.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["Exited"])

def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], str]:
    """
    Retrieves input feature columns and target column.

    :param df: The DataFrame containing features and target.
    :return: A tuple containing a list of feature column names and the target column name.
    """
    input_cols = list(df.columns)[1:-1]
    target_col = 'Exited'
    return input_cols, target_col

def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list]:
    """
    Encodes categorical features using OneHotEncoder.

    :param train_df: Training DataFrame.
    :param val_df: Validation DataFrame.
    :param categorical_cols: List of categorical columns to encode.
    :return: Tuple containing transformed train_df, val_df, the fitted encoder, and the encoded column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])

    train_df.drop(columns=categorical_cols, inplace=True)
    val_df.drop(columns=categorical_cols, inplace=True)

    return train_df, val_df, encoder, encoded_cols

def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: list) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numeric features using MinMaxScaler.

    :param train_df: Training DataFrame.
    :param val_df: Validation DataFrame.
    :param numeric_cols: List of numeric columns to scale.
    :return: Tuple containing transformed train_df, val_df, and the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains a logistic regression model.

    :param X_train: Training feature set.
    :param y_train: Training target values.
    :return: Trained logistic regression model.
    """
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model

def process_data(clients_df: pd.DataFrame, scaler_numeric: bool) -> dict:
    """
    Main function to process data including feature engineering, scaling, encoding, and model training.

    :param clients_df: The raw DataFrame containing client data.
    :return: Dictionary containing processed data and trained components.
    """
    # Drop unnecessary columns
    clients_df = drop_unnecessary_columns(clients_df, ['CustomerId', 'Surname'])

    # Split dataset into train and validation
    train_df, val_df = split_data(clients_df)

    # Extract input and target columns
    input_cols, target_col = get_feature_columns(train_df)

    # Separate features and targets
    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    # Identify categorical and numeric columns
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    # Encode categorical features
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    # Convert specific columns to integer
    int_cols = ['Age', 'Tenure', 'NumOfProducts', 'HasCrCard']
    for col in int_cols:
        if col in train_inputs:
            train_inputs[col] = train_inputs[col].astype(int)
            val_inputs[col] = val_inputs[col].astype(int)

    # Scale numeric features
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    else:
        train_inputs, val_inputs = train_inputs, val_inputs
        scaler = None

    # Train model
    model = train_model(train_inputs[numeric_cols + encoded_cols], train_targets)

    return {
        'X_train': train_inputs[numeric_cols + encoded_cols],
        'train_targets': train_targets,
        'X_val': val_inputs[numeric_cols + encoded_cols],
        'val_targets': val_targets,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder,
        'model': model
    }

def process_new_data(raw_df: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    raw_df = drop_unnecessary_columns(raw_df, ['id', 'CustomerId', 'Surname'])

    int_cols = ['Age', 'Tenure', 'NumOfProducts', 'HasCrCard']
    for col in int_cols:
        if col in raw_df:
            raw_df[col] = raw_df[col].astype(int)
            raw_df[col] = raw_df[col].astype(int)

    numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in {'id', 'Exited'}]
    categorical_cols = raw_df.select_dtypes(include='object').columns.tolist()

    # Scale numeric features
    if numeric_cols:
        transformed_numeric = scaler.transform(raw_df[numeric_cols])
        raw_df[numeric_cols] = transformed_numeric  # ✅ Ensure same shape

    # Encode categorical features
    if categorical_cols:
        transformed_categorical = encoder.transform(raw_df[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        transformed_categorical_df = pd.DataFrame(transformed_categorical, columns=encoded_cols, index=raw_df.index)
        raw_df = pd.concat([raw_df.drop(columns=categorical_cols), transformed_categorical_df], axis=1)

    return raw_df
