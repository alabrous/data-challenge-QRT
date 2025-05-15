import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest


def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"File {file_path} is empty.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def preprocess_data(df,target_df):
    """
    Preprocess the DataFrame by handling missing values and converting data types.
    """

    # Drop rows where 'OS_YEARS' is NaN if conversion caused any issues
    target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)    

    # Contarget_dfvert 'OS_YEARS' to numeric if it isn’t already
    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')

    # Ensure 'OS_STATUS' is boolean
    target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

    y = Surv.from_dataframe('OS_STATUS', 'OS_YEARS', target_df)

    features  = ['BM_BLAST','WBC','ANC','MONOCYTES','HB','PLT','CYTOGENETICS']

    # Apply Yeo Johnson transformation to normalize the data
    for column in ['BM_BLAST','WBC','ANC','HB','PLT']:
        if column in df.columns:
            transformer = PowerTransformer(method='yeo-johnson')
            df[column] = transformer.fit_transform(df[[column]])
        else:
            print(f"Column {column} not found in DataFrame.")

    X = df.loc[df['ID'].isin(target_df['ID']), features]

    X = extract_cytogenetic_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    imputer = SimpleImputer(strategy="median")
    X_train[['BM_BLAST', 'HB', 'PLT']] = imputer.fit_transform(X_train[['BM_BLAST', 'HB', 'PLT']])
    X_test[['BM_BLAST', 'HB', 'PLT']] = imputer.transform(X_test[['BM_BLAST', 'HB', 'PLT']])

    return X_train, X_test, y_train, y_test

def extract_cytogenetic_features(CYTOGENETICS_df, anomaly_patterns=None):
    """
    Extract cytogenetic anomalies from text data and convert to binary features.
    
    Args:
        CYTOGENETICS_df (pd.DataFrame): DataFrame containing CYTOGENETICS column
        anomaly_patterns (dict, optional): Dictionary of anomaly patterns to search for
        
    Returns:
        pd.DataFrame: DataFrame with binary columns for each anomaly
    """
    if anomaly_patterns is None:
        # Common cytogenetic abnormalities in MDS/AML
        anomaly_patterns = {
            'del5q': [r'del.*5q', r'del.*\(5\)', r'5q-', r'-5'],
            'del7q': [r'del.*7q', r'del.*\(7\)', r'7q-', r'-7'],
            'trisomy8': [r'\+8', r'trisomy.*8'],
            'del17p': [r'del.*17p', r'del.*\(17\)', r'17p-'],
            'del20q': [r'del.*20q', r'del.*\(20\)', r'20q-'],
            'monosomy7': [r'-7', r'monosomy.*7'],
            'complex': [r'complex'],
            'normal': [r'normal', r'nn', r'46,xx', r'46,xy'],
            'inv3': [r'inv.*3', r'inv.*\(3\)'],
            't8_21': [r't\(8;21\)'],
            'inv16': [r'inv.*16', r'inv.*\(16\)']
        }
    
    # Create copy of input DataFrame
    df = CYTOGENETICS_df.copy()
    
    # Convert CYTOGENETICS column to lowercase
    df['CYTOGENETICS'] = df['CYTOGENETICS'].str.lower()
    
    # Create binary columns for each anomaly
    for anomaly, patterns in anomaly_patterns.items():
        pattern = '|'.join(patterns)
        df[f'cyto_{anomaly}'] = df['CYTOGENETICS'].str.contains(
            pattern, 
            case=False, 
            regex=True,
            na=False
        ).astype(int)
    
    # Count total anomalies (excluding 'normal' and 'complex')
    exclude_cols = ['cyto_normal', 'cyto_complex']
    anomaly_cols = [col for col in df.columns if col.startswith('cyto_') 
                   and col not in exclude_cols]
    df['total_anomalies'] = df[anomaly_cols].sum(axis=1)

    # Drop original CYTOGENETICS column
    df.drop(columns=['CYTOGENETICS'], inplace=True)
    # Drop columns for each anomaly
    for anomaly in anomaly_patterns.keys():
        if anomaly != 'normal' and anomaly != 'complex':
            df.drop(columns=[f'cyto_{anomaly}'], inplace=True)
    
    return df

def preprocess_main(clinical_file_path,target_file_path):
    df = load_data(clinical_file_path)
    if df is None:
        return
    
    target_df = load_data(target_file_path)
    if target_df is None:
        return
    
    X_train, X_test, y_train, y_test = preprocess_data(df,target_df)
    
    return X_train, X_test, y_train, y_test


    