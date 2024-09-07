# Cancellation Fraud Detection in Ride-Hailing Services Using Machine Learning

This project aims to detect fraudulent behavior in ride-hailing services, where drivers accept trips but cancel them to avoid paying commission. The notebook leverages machine learning techniques to identify such drivers.

## Overview

The project employs a variety of machine learning algorithms to predict fraudulent trip cancellations. We address the imbalanced dataset problem and apply techniques such as oversampling and detailed hyperparameter tuning.

### Key Features:
1. **Data Preprocessing**: Handling imbalanced data using SMOTE and ADASYN.
2. **Machine Learning Pipelines**: Several classifiers like Random Forest, SVM, and KNN are tuned using grid search.
3. **Outlier Detection**: Z-score based filtering is applied to remove extreme values from the dataset.
4. **Feature Correlation**: A dendrogram is used to visualize hierarchical clustering of dataset features.

## Dependencies

- Python 3.x
- `scikit-learn`
- `imblearn`
- `pandas`
- `plotly`
- `matplotlib`

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/smasoudrezvani/cancellation_fraud.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to preprocess data, train models, and visualize results.

## Key Sections in the Notebook

1. **Library Setup**: Imports necessary libraries for machine learning and data handling.
2. **Data Preprocessing**: Z-score based outlier removal and dataset cleaning.
3. **Modeling**: Hyperparameter tuning using Grid Search with Random Forest, SVM, and KNN.
4. **Visualization**: Plotting interactive dendrograms to understand feature correlations.

