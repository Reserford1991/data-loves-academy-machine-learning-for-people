import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from ydata_profiling import ProfileReport
from typing import Tuple, Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from scipy.stats import shapiro
import warnings
warnings.filterwarnings("ignore")

class HelperFunctions:
    def __init__(self):
        return

    @staticmethod
    def write_ydata_report_to_file(df: pd.DataFrame, filename: str) -> None:
        """
        Function to write ydata report to file.

        :param df: The raw DataFrame containing data.
        :param filename: The filename to save the report to.

        :return: None
        """

        ydata_report = ProfileReport(df)
        ydata_report.to_file(filename)

    @staticmethod
    def show_numeric_info( df: pd.DataFrame, column_name: str, n_bins: int) -> None:
        """
        Function to write ydata report to file.

        :param df: The raw DataFrame containing data.
        :param column_name: The column name to save the report to.
        :param n_bins: The number of bins to use for the histogram.

        :return: None
        """

        print(df[column_name].describe())

        print('Number of null values:', df[column_name].isnull().sum())

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        sns.boxplot(x=df[column_name], ax=axes[0], color='blue')
        axes[0].set_title(f'Box plot of {column_name}', fontsize=16, fontweight='bold')

        axes[1].hist(df[column_name], bins=n_bins, color='blue', edgecolor='black')
        axes[1].set_xlabel(column_name, fontsize=14)
        axes[1].set_ylabel('Frequency', fontsize=14)
        axes[1].set_title(f'Histogram of {column_name}', fontsize=16, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_categorical_info(df: pd.DataFrame, column_name: str) -> None:
        """
        Function to show categorical column information.

        :param df: The DataFrame containing data.
        :param column_name: The categorical column name to analyze.

        :return: None
        """

        print(df[column_name].describe())

        print('Number of null values:', df[column_name].isnull().sum())

        # Count occurrences of each category
        counts = df[column_name].value_counts()

        # Set figure size
        plt.figure(figsize=(12, 6), facecolor="black")

        # Create horizontal bar chart
        ax = sns.barplot(y=counts.index, x=counts.values, color='royalblue')
        ax.set_facecolor("black")

        # Adjust text labels inside bars
        for index, value in enumerate(counts.values):
            ax.text(min(value * 0.05, max(counts.values)), index, str(value),
                    color="white", fontsize=14, fontweight='bold', va='center')

        # Add a title to the plot
        plt.title(f"Distribution of {column_name} categories", fontsize=16, fontweight='bold', color='white')
        # Improve axis visibility
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks(fontsize=14, color='white')  # Make category labels bigger
        plt.grid(axis="x", linestyle="--", alpha=0.5)  # Add subtle grid lines

        # Remove borders
        sns.despine(left=True, bottom=True)

        # Show plot
        plt.show()

        print(df[column_name].dtype)

    @staticmethod
    def impute_categorical_unknown_values(df: pd.DataFrame, column_name: str, strategy: str) -> pd.DataFrame:
        """
        This function imputes categorical columns with unknown values.

        :param df: The raw DataFrame containing data.
        :param column_name: The column name to impute.

        :return: The imputed DataFrame.
        """
        df[column_name] = df[column_name].replace('unknown', np.nan)

        match strategy:
            case "mode":
                mode = df[column_name].mode()[0]
                df[column_name].fillna(mode, inplace=True)

            case _:
                print (f"[INFO] invalid imputation strategy '{strategy}'. No imputation will be performed. Supported strategies: ['mode']")

        return df

    @staticmethod
    def transform_string_into_category_type(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
        """
        This function transforms categorical column into a categorical column.

        :param df:
        :param column_names:

        :return: transformed DataFrame.
        """

        for column_name in column_names:
            if column_name in df.columns:
                df[column_name] = df[column_name].astype('category')

        return df


    @staticmethod
    def transform_right_skewed_numerical_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        This function transforms categorical column into a categorical column.

        :param df: raw DataFrame containing data.
        :param column_name: name of the column to transform.

        :return: transformed DataFrame.
        """


        new_column_name = f"{column_name}_capped"
        threshold = df[column_name].quantile(0.99)
        df[new_column_name] = np.where(df[column_name] >= threshold, threshold, df[column_name])
        return df

    @staticmethod
    def handle_outliers(df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Function to handle outliers

        :param df: raw DataFrame containing data.
        :param numerical_cols: list of numerical columns.

        :return: transformed DataFrame.
        """
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        return df

    @staticmethod
    def scale_features(df: pd.DataFrame, numerical_cols: List[str], threshold: float = 0.05, use_stat: bool = True) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Function to scale numerical columns.

        :param df: raw DataFrame containing data.
        :param numerical_cols:
        :param threshold:
        :param use_stat:
        :return:
        """
        df_scaled = df.copy()
        min_max_scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        min_max_cols = []
        standard_cols = []

        if use_stat:
            for col in numerical_cols:
                ## shapiro test for normality
                stat, p = shapiro(df[col].sample(500) if len(df[col]) > 500 else df[col])

                ## If normally distributed - use StandardScaler
                if p > threshold:
                    df_scaled[col] = standard_scaler.fit_transform(df[col].values.reshape(-1, 1))
                    standard_cols.append(col)
                ## Otherwise use MinMaxScaler
                else:
                    df_scaled[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))
                    min_max_cols.append(col)
        else:
            for col in numerical_cols:
                df_scaled[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))
                min_max_cols.append(col)

        return df_scaled, {"min_max_scaled_cols": min_max_cols, "standard_scaled_cols": standard_cols}

    @staticmethod
    def process_data_for_linear_regression(df: pd.DataFrame, target_col) -> tuple[DataFrame, Any, dict[str, list[str]]]:
        """
         Function to process the raw data for linear regression.

        :param df:
        :param target_col:
        :return: processed DataFrame.
        """
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        # 1. Handle outliers
        df = HelperFunctions.handle_outliers(df, numeric_cols)

        # 2. Scale features
        df_scaled, scaler_info = HelperFunctions.scale_features(df, numeric_cols)

        # 3. One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(df[categorical_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols),
                                      index=df.index)

        # 4. Combine numerical and encoded categorical features
        processed_df = pd.concat([df_scaled[numeric_cols], encoded_cat_df], axis=1)

        # 5. Separate target column
        target = df_scaled[target_col]

        return processed_df, target, scaler_info

    @staticmethod
    def process_data_for_knn_klassification(df: pd.DataFrame, target_col) -> tuple[DataFrame, Any, dict[str, list[str]]]:
        """
        Function to process the raw data for knn klassification.

        :param df:
        :param target_col:
        :return: processed DataFrame.
        """
        # 1. Preprocess
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        # 2. Outlier Handling
        bank_df = HelperFunctions.handle_outliers(df, numeric_cols)

        # 3. Feature Scaling
        bank_df_scaled, scaler_info = HelperFunctions.scale_features(bank_df, numeric_cols, False)

        # 4. One-hot Encoding for categorical columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(bank_df[categorical_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols),
                                      index=bank_df.index)

        # 5. Combine numerical and encoded categorical features
        processed_df = pd.concat([bank_df_scaled[numeric_cols], encoded_cat_df], axis=1)

        # 6. Separate target column
        target = bank_df[target_col]

        return processed_df, target, scaler_info

    @staticmethod
    def preprocess_for_decision_tree(df: pd.DataFrame, target_col) -> tuple[DataFrame, Any]:
        """
        Function to preprocess the raw data for decision tree.

        :param df: input DataFrame
        :param target_col: name of the target column
        :return: Tuple (X, y)
        """

        number_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col in number_cols:
            number_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cat = encoder.fit_transform(df[categorical_cols])
        encoded_cat_df = pd.DataFrame(
            encoded_cat,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )

        X = pd.concat([df[number_cols], encoded_cat_df], axis=1)
        y = df[target_col]

        return X, y

    @staticmethod
    def impute_numeric_data(df: pd.DataFrame, col_name: str, strategy: str, constant: int = None) -> pd.DataFrame:
        """
        Function to impute numeric data.

        :param df: pd.DataFrame
        :param col_name: str
        :param strategy: str
        :param constant: int - only required if strategy == "constant"

        :return: df:pd.DataFrame
        """

        match strategy:
            case "median":
                df[col_name].fillna(df[col_name].median(), inplace=True)
            case "mean":
                df[col_name].fillna(df[col_name].mean(), inplace=True)
            case "constant":
                df[col_name].fillna(constant, inplace=True)
            case _:
                print(
                    f"[INFO] invalid imputation strategy '{strategy}'. No imputation will be performed. Supported strategies: ['median', 'mean', 'constant']")


        return df

    @staticmethod
    def process_categorical_data_for_clusterization(df: pd.DataFrame, col_name: str, strategy: str) -> pd.DataFrame:
        """
        Function to process the raw data for clusterization.

        :param df: pd.DataFrame
        :param strategy: str
        :return: col_name: str

        :return: pd.DataFrame
        """

        match strategy:
            case "low-cardinality":
                df = pd.concat([
                    df.drop(columns=[col_name]),
                    pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)

                ], axis=1)
            case "mid-high-cardinality":
                freq_map = df[col_name].value_counts(normalize=True)
                df[col_name + "_freq"] = df[col_name].map(freq_map)
                df.drop(columns=[col_name], inplace=True)
            case _:
                print(
                    f"[INFO] invalid imputation strategy '{strategy}'. No imputation will be performed. Supported strategies: ['low-cardinality', 'mid-high-cardinality']")

        return df

    @staticmethod
    def process_date_for_clusterization(df: pd.DataFrame, col_name: str, date_format: str) -> pd.DataFrame:
        """
        Function to process date column for clusterization.

        :param df: pd.DataFrame
        :param col_name: str
        :param date_format: str

        :return: pd.DataFrame
        """

        df[col_name] = pd.to_datetime(df[col_name], format=date_format)

        # Example features
        df['days_since_registration'] = (pd.Timestamp("today") - df['Dt_Customer']).dt.days
        df['registration_year'] = df['Dt_Customer'].dt.year
        df['registration_month'] = df['Dt_Customer'].dt.month
        df.drop(columns=['Dt_Customer'], inplace=True)

        return df