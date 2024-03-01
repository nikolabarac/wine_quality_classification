# Wine Quality Classification Project

This project involves exploring and classifying the quality of red wine based on various features. The dataset used in this project is stored in the 'winequality-red.csv' file.

## Exploratory Data Analysis

The initial exploration of the dataset includes visualizing histograms for each column to understand the distribution of the data.

A correlation matrix and pair plot are also generated to analyze the relationships between different features.

Box plots and kernel density plots are created to further investigate the distribution and outliers of each feature, both by wine quality and class.

## Data Preprocessing

The dataset is loaded and preprocessed to create a balanced train and test set. The 'class' column is derived from the 'quality' column, where wines with a quality greater than 5.5 are labeled as class 1 and the rest as class 0.

## Model Selection and Evaluation

A pipeline is implemented to standardize features and evaluate various classification algorithms using cross-validation. The following classifiers are included in the comparison:

- RandomForestClassifier
- GradientBoostingClassifier
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Decision Tree

The performance of each classifier is assessed based on accuracy, precision, recall, and ROC AUC. The ROC curves are also plotted for visual representation.

## Hyperparameter Tuning

The best-performing algorithm is identified, and its hyperparameters are fine-tuned using GridSearchCV.

## Model Testing and Export

The tuned model (RandomForest) is tested on the test set, and its performance is evaluated. Finally, the trained model is exported using joblib.

Feel free to adapt and run the provided code snippets in your Python environment.
