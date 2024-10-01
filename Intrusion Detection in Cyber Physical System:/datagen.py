from load_save import save, load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD


def extract_static_features(dataset):
    # Placeholder for static feature extraction
    # You can replace this with your actual logic
    dataset['file_size'] = np.random.randint(1000, 10000, dataset.shape[0])  # Mock file size
    dataset['file_type'] = np.random.choice(['exe', 'dll', 'txt'], dataset.shape[0])  # Mock file types
    dataset['entropy'] = np.random.uniform(0, 8, dataset.shape[0])  # Mock entropy
    return dataset


def extract_dynamic_features(dataset):
    # Placeholder for dynamic feature extraction
    # You can replace this with your actual logic
    dataset['api_calls'] = np.random.randint(1, 100, dataset.shape[0])  # Mock API calls
    dataset['system_calls'] = np.random.randint(1, 50, dataset.shape[0])  # Mock system calls
    dataset['network_traffic'] = np.random.uniform(0, 1000, dataset.shape[0])  # Mock network traffic patterns
    return dataset


def datagen():
    # Load the dataset
    dataset = pd.read_csv("dataset/kdd_dataset.csv")

    # Drop unnecessary columns
    drop = ['Unnamed: 0', 'num_outbound_cmds']
    dataset = dataset.drop(columns=drop)

    # Class mapping
    class_mapping = {
        "back": 0, "land": 0, "neptune": 0, "pod": 0, "smurf": 0, "teardrop": 0,
        'apache2': 0, 'processtable': 0, 'worm': 0,
        "buffer_overflow": 1, "perl": 1, "loadmodule": 1, "rootkit": 1,
        'ps': 1, 'httptunnel': 1, 'mailbomb': 1, 'xlock': 1, 'xsnoop': 1,
        "ftp_write": 2, "guess_passwd": 2, "imap": 2, "multihop": 2,
        'named': 1, 'sendmail': 1, 'xterm': 1, 'sqlattack': 1, 'udpstorm': 1,
        "phf": 2, "spy": 2, "warezclient": 2, "warezmaster": 2,
        'snmpguess': 2, 'snmpgetattack': 2,
        "ipsweep": 3, "nmap": 3, "portsweep": 3, "satan": 3,
        'mscan': 3, 'saint': 3,
        "normal": 4,
    }

    # Replace values in the 'class' column based on the class_mapping dictionary
    dataset['class'] = dataset['class'].replace(class_mapping)

    # Drop rows with NaN values (if any remain after mapping)
    dataset = dataset.dropna()

    # Hot-deck imputation for missing values (using KNN)
    imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors
    dataset_imputed = imputer.fit_transform(dataset)
    dataset = pd.DataFrame(dataset_imputed, columns=dataset.columns)

    # Outlier detection using Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=2)  # Adjust based on your data
    dataset['outliers'] = gmm.fit_predict(dataset)

    # Remove outliers (assuming class 1 corresponds to outliers)
    dataset = dataset[dataset['outliers'] == 0].drop(columns='outliers')

    # Normalization using Z-score normalization
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset.drop(columns=['class']))
    dataset_scaled = pd.DataFrame(dataset_scaled, columns=dataset.columns[:-1])

    # Re-add the class column to the scaled dataset
    dataset_scaled['class'] = dataset['class'].reset_index(drop=True)

    # Feature extraction
    dataset_scaled = extract_static_features(dataset_scaled)
    dataset_scaled = extract_dynamic_features(dataset_scaled)

    # Dimensionality Reduction using SVD
    features = dataset_scaled.drop(columns=['class'])
    svd = TruncatedSVD(n_components=10)  # Choose number of components based on your needs
    features_reduced = svd.fit_transform(features)

    # Convert reduced features back to DataFrame
    reduced_features_df = pd.DataFrame(features_reduced,
                                       columns=[f'feature_{i + 1}' for i in range(features_reduced.shape[1])])

    # Re-add the class column to the reduced features DataFrame
    reduced_features_df['class'] = dataset_scaled['class'].reset_index(drop=True)

    # Final dataset after cleaning, normalization, feature extraction, and dimensionality reduction
    print(reduced_features_df)
    labels = np.array(dataset['class'])


    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(reduced_features_df, labels, test_size=0.2, random_state=42)
    save('X_train', X_train)
    save('X_test', X_test)
    save('y_train', y_train)
    save('y_test', y_test)

#
# # Call the function
datagen()
