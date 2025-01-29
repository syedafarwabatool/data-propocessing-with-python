import joblib
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import zscore

st.set_page_config(
    page_title="Data Preprocessing Pipeline",
    page_icon="🦩",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file_path):
    return pd.read_csv(file_path)

def identify_columns(df):
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical, categorical

def handle_missing_values(df, strategy="mean"):
    imputer = SimpleImputer(strategy=strategy)
    df[df.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df.select_dtypes(include=[np.number]))
    return df

def identify_outliers(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(zscore(df[numerical_cols]))
    return (z_scores > 3).any(axis=1)

def cap_outliers(df, method='zscore', threshold=3):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'zscore':
        # Calculate Z-scores for numerical columns
        z_scores = zscore(df[numerical_cols])
        for col, z in zip(numerical_cols, z_scores.T):
            # Compute thresholds
            lower_limit = df[col].mean() - threshold * df[col].std()
            upper_limit = df[col].mean() + threshold * df[col].std()
            # Cap outliers
            df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    elif method == 'iqr':
        # Calculate IQR (Interquartile Range) for numerical columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - threshold * IQR
            upper_limit = Q3 + threshold * IQR
            # Cap outliers
            df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    else:
        raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")
    
    return df

# Function to plot outliers using boxplots with subplots
def plot_outliers(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = len(numerical_cols)
    
    # Define a grid size for subplots based on the number of numerical columns
    ncols = 6  # Number of columns in the subplot grid (you can adjust this)
    nrows = (num_cols // ncols) + (1 if num_cols % ncols != 0 else 0)  # Calculate rows needed
    
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()  # Flatten the axes array to make it easy to iterate

    for i, col in enumerate(numerical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Outliers in {col}")
    
    # Remove empty subplots (if any)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # Display the plots
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def encode_features(df):
    encoder = OneHotEncoder()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    encoded_df = pd.DataFrame(encoder.fit_transform(df[cat_cols]).toarray(), columns=encoder.get_feature_names_out())
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    return df

def scale_features(df):
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def apply_pca(df, threshold=0.95):
    
    # Select numerical columns only
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) == 0:
        raise ValueError("No numerical columns available for PCA.")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Fit PCA on the numerical data
    pca = PCA()
    pca.fit(df[numerical_cols])
    
    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Ensure cumulative_variance is purely numeric
    if not np.issubdtype(cumulative_variance.dtype, np.number):
        raise ValueError("Cumulative variance contains non-numeric data.")
    
    # Determine the optimal number of components
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1  # First index where threshold is met
    
    # Apply PCA with the optimal number of components
    pca = PCA(n_components=optimal_components)
    pca_result = pca.fit_transform(df[numerical_cols])
    
    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(optimal_components)], index=df.index)
    
    # Combine PCA-transformed columns with non-numerical columns
    non_numerical_cols = df.select_dtypes(exclude=[np.number]).columns
    final_df = pd.concat([df[non_numerical_cols], pca_df], axis=1)
    
    return final_df

# Streamlit App
st.title("Data Preprocessing Pipeline")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Check if file is uploaded and initialize session state
if uploaded_file is not None:
    if 'df' not in st.session_state:
        st.session_state.df = load_data(uploaded_file)
    st.write("### Uploaded Dataset", st.session_state.df)

    if st.button("Identify Numerical and Categorical Fields"):
        numerical, categorical = identify_columns(st.session_state.df)
        st.write("Numerical Columns:", numerical)
        st.write("Categorical Columns:", categorical)

    # Step 1: User selects the method to remove outliers
    method = st.selectbox(
        "Select the method to remove outliers:",
        ("zscore", "iqr")  # Options for the user
    )

    # Step 2: Confirm the choice with a button
    if st.button("Confirm Outlier Removal Method"):
        st.session_state.outlier_method = method  # Save the method in session state
        st.success(f"Outlier removal method selected: {method}")

    # Step 3: Identify and remove outliers only after method is selected and confirmed
    if "outlier_method" in st.session_state:
        if st.button("Identify and Remove Outliers"):
            # Show the number of outliers detected
            st.write("### Outliers Detected:", identify_outliers(st.session_state.df).sum())
            
            # Plot initial outliers
            plot_outliers(st.session_state.df)
            
            # Remove or cap outliers using the selected method
            st.session_state.df = cap_outliers(st.session_state.df, method=st.session_state.outlier_method)
            
            # Display the dataset after removing outliers
            st.write("### Dataset After Removing Outliers", st.session_state.df)

    if st.button("Handle Missing Values"):
        st.session_state.df = handle_missing_values(st.session_state.df, strategy="mean")
        st.write("### Dataset After Handling Missing Values", st.session_state.df)

    threshold = st.selectbox(
        "Select the threshold for variance in PCA :",
        ("0.85","0.9", "0.95")  # Options for the user
    )

    if st.button("Confirm threshold"):
        st.session_state.PCA_threshold = float(threshold)  # Save the method in session state
        st.success(f"Threshold selected: {threshold}")

    if "PCA_threshold" in st.session_state:

        if st.button("Apply PCA"):
            st.session_state.df = apply_pca(st.session_state.df,st.session_state.PCA_threshold)
            st.write("### Dataset After Applying PCA", st.session_state.df)

    if st.button("Encode Features"):
        st.session_state.df = encode_features(st.session_state.df)
        st.write("### Dataset After Encoding Features", st.session_state.df)

    if st.button("Scale Features"):
        st.session_state.df = scale_features(st.session_state.df)
        st.write("### Dataset After Scaling Features", st.session_state.df)

    if st.button("Generate Preprocessing Pipeline"):
        # Create the pipeline
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('scaler', StandardScaler()),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Display the pipeline in a readable format
        st.write("### Pipeline Generated")
        st.json({
            "steps": [step[0] for step in pipeline.steps],
            "details": [str(step[1]) for step in pipeline.steps]
        })

        # Serialize the pipeline using joblib
        pipeline_file = BytesIO()
        joblib.dump(pipeline, pipeline_file)
        pipeline_file.seek(0)

        # Create a download button for the pipeline
        st.download_button(
            label="Download Pipeline",
            data=pipeline_file,
            file_name="preprocessing_pipeline.joblib",
            mime="application/octet-stream"
        )