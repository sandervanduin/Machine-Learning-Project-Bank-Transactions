# Machine-Learning-Project-Bank-Transactions
This project was made during my Master Digital Driven Business while following the course AI Methods for Businesses

# Bank Transaction Customer Segmentation

## Project Description
This project, as part of a master's program in digital-driven business, entails customer segmentation using a bank transactions dataset. It aims to identify high-value customers by employing various clustering techniques such as K-means, DBSCAN, and HDBSCAN.
___
## Authors
- `Sander van Duin`
- `Ali Kalantari Khandani`
- `Owen Alberts`
- `Patrick Nasr-Alla`
___
## Why This Project?
- **Educational Goal**: Applying predictive tools and clustering techniques to a real-world dataset.
- **Business Relevance**: Clustering customers in the banking sector to understand customer behavior and value.

## What Makes This Project Stand Out?
- **Methodological Diversity**: Employs and compares multiple clustering techniques, providing a broad perspective on data segmentation methods.

## Files

- Jupyter Notebooks
    - EDA
        - EDA Bank Transaction (1)_markdowns.ipynb
    - Models
        - Final Model with all Variables.ipynb
        - Final Model with all Variables except MMR.ipynb
        - Final Model with body, 'condition', 'odometer', 'mmr', 'car_age', 'coa_score'.ipynb
        - Final Model with MMR and Numerical Features.ipynb
        - Final Model with only MMR.ipynb
- CSV files
    - EDA
        - 
    - Models
        - 
- Results
    - Results.ipynb

Before beginning to work with the notebooks in this project, it's important to set up your working environment correctly. This involves specifying the path to the directory containing your datasets and ensuring that all necessary libraries are installed. Here's a step-by-step guide to setting up your environment:

# Import necessary libraries
import os
import pandas as pd

# Set the path to the directory containing the datasets
path = "Your Pathname"  # Replace with the path to your dataset directory

# Change the current working directory to the specified path
os.chdir(path)

# List files in the 'bank' subdirectory to verify the presence of the dataset
os.listdir(os.path.join('bank'))

# Construct the file paths for the datasets
bank_transaction_path = os.path.join("bank", "modified_bank_transactions.csv")
bank_df_Original = pd.read_csv(bank_transaction_path, sep=',')

# Load the datasets into pandas DataFrames
# You may need to adjust the loading parameters based on the structure of your CSV files
bank_transactions_df = pd.read_csv(bank_transactions_path)

# Display the first few rows of the DataFrames for a preliminary look at the data
print(bank_transactions_df.head())

## Table of Contents
1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
2. [Feature Engineering](#2-feature-engineering)
3. [Cluster Analysis](#3-cluster-analysis)
4. [Detailed Insights and Subquestions](#4-detailed-insights-and-subquestions)


## 1. Exploratory Data Analysis (EDA)
### Overview
- In-depth analysis of the dataset including data cleaning, handling missing values, and visualizations.
- Exploration of transaction amounts, account balances, and customer demographics.

### Objectives

## Project Objectives
 **Customer Clustering**:
   - Segment customers using K-means, DBSCAN, and HDBSCAN.
   - Evaluate and select the most effective clustering method.

 **Dimensionality Reduction**:
   - Apply PCA to enhance clustering model performance.

 **Comparative Analysis**:
   - Contrast different clustering techniques to identify the best fit for the dataset.

 **Insight Generation**:
   - Derive actionable insights for business strategy.
   - Identify high valuable customer segments.

### Data Sources
- Primary Dataset: `modified_bank_transactions.csv` containing detailed transaction records.
- Supplementary Dataset: `india_cities.csv` providing additional customer demographics.

## Tools and Techniques

- **Data Manipulation and Analysis**:
  - `pandas`: For handling and manipulating the dataset.
  - `numpy`: For numerical computations and array manipulations.

- **Data Visualization**:
  - `matplotlib.pyplot` and `seaborn`: For creating a wide range of static, interactive, and informative visualizations.
  - `yellowbrick.cluster`: Specifically for the KElbowVisualizer to help determine the optimal number of clusters.
  - `mpl_toolkits.mplot3d`: For 3D plotting capabilities.

- **Data Preprocessing and Feature Engineering**:
  - `sklearn.compose.ColumnTransformer` and `sklearn.pipeline.Pipeline`: For streamlined preprocessing workflows.
  - `sklearn.impute.KNNImputer`: For imputing missing data using k-Nearest Neighbors.
  - `sklearn.preprocessing.StandardScaler`: For feature scaling to standardize the dataset.

- **Clustering and Dimensionality Reduction**:
  - `sklearn.decomposition.PCA`: For Principal Component Analysis to reduce dimensionality.
  - `sklearn.cluster.KMeans`: For applying K-means clustering algorithm.
  - `sklearn.cluster.DBSCAN`: For Density-Based Spatial Clustering of Applications with Noise.
  - `hdbscan`: A high performance implementation of HDBSCAN clustering algorithm.

- **Miscellaneous**:
  - `scipy.stats`: For statistical testing and operations.
  - `sklearn.metrics`: For evaluating the clustering models.
  - `sklearn.datasets.make_blobs`: For generating sample data points.
  - `sklearn.neighbors.NearestNeighbors`: For nearest neighbor queries, used in DBSCAN.
  - `matplotlib.colors.ListedColormap`: For custom color maps in visualizations.


## 2. Feature Engineering
### Process
- Gender encoding, monetary conversion from INR to Euro, and data cleaning for missing values and outliers. 

Extra features:
Converting transactiondate to datetime
Calculating recency/frequency/monetary
Calculating age and average account balance 
Using the top 10 city’s as features

## 3. Cluster Analysis
### 3.1 K-means and PCA
- Application of K-means clustering followed by PCA for dimensionality reduction.

### 3.2 DBSCAN and HDBSCAN
- Utilization of DBSCAN and HDBSCAN for comparative cluster analysis.
- Examination of PCA components in cluster formation.

## 4. Detailed Insights and Subquestions
### Insights
- Identification of distinct customer segments and their transaction behaviors.
- Analysis of customer activity patterns and transaction types.

### Researchquestion
How can insights from the dataset be used to identify and better understand high-value customers, enabling the bank to tailor its financial products and services more effectively?

### Subquestions
- How Can High-Value Customers Be Identified?
- How Can We Understand the Spending Patterns of High-Value Customers?
- How Can Services Be Customized for High-Value Customers?
- How Can Risk Be Assessed for High-Value Customers?
- How Can Fraud Detection Be Enhanced Among High-Value Customers?


## 5. Reflection
Reflecting on this project, it's been a truly enlightening team effort. We learned the critical importance of grasping the foundational principles behind machine learning and testing. Before we could confidently use all the methods to our research questions, we needed to thoroughly understand which models were appropriate under which conditions, ensuring that our choices were not just correct, but also relevant to our specific data.

Creating a custom tool for number crunching turned out to be a pivotal element in our project. This tool greatly simplified our analysis of the data, which was quite extensive, and it significantly enhanced our collective understanding of the intricate relationships within our dataset. The development process was challenging, but it was a lesson to our collective commitment to precision and our analytical skills.

One of the major challenges we faced as a team was the volume of data. It was an ambitious endeavor that, in retrospect, could have been approached more efficiently. In future projects, we'll consider using a smaller subset of the data—perhaps 10%—from the start. This would likely save us time and computational resources while still providing an accurate picture for our analysis.

Each hurdle we encountered was a learning opportunity for us as a team. There were moments when managing such a large dataset felt daunting. However, each challenge pushed us to think more creatively and to refine our methods for the better.

Reflecting on this project, we've truly come to appreciate the complexities and intricacies of machine learning and statistical testing. As a team, we learned the critical importance of understanding the foundational principles of these methodologies. Before applying various models to our research questions, we immersed ourselves in comprehending which approaches were most appropriate for our specific data. This wasn't just about correctness but relevance and suitability to the unique characteristics of our dataset.

Creating a custom tool for data analysis turned out to be a key milestone in our project. This tool significantly streamlined our analysis, which was crucial given the extensive nature of our data. More than just a technical achievement, this development process was a testament to our collective commitment to precision and a reflection of our growing analytical capabilities.
