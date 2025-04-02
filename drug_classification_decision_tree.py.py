# Import necessary libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)

# Display dataset information
print("Dataset Information:")
print(my_data.info())

# Check for missing values
print("\nMissing Values in Dataset:")
print(my_data.isnull().sum())

# Data Analysis and Preprocessing
print("\nEncoding Categorical Variables...")

# Encode categorical features
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

# Encode the target variable (Drug)
my_data['Drug'] = label_encoder.fit_transform(my_data['Drug'])

# Display the first few rows of the transformed dataset
print("\nTransformed Dataset:")
print(my_data.head())

# Verify no missing values remain
print("\nMissing Values After Transformation:")
print(my_data.isnull().sum())

# Correlation Analysis
print("\nCorrelation Analysis:")
my_data['Drug_num'] = my_data['Drug']  # Map Drug to numerical values for correlation
correlation_values = my_data.drop('Drug', axis=1).corr()['Drug_num'].sort_values(ascending=False)
print(correlation_values)

# Visualize Category Distribution
print("\nVisualizing Category Distribution...")
category_counts = my_data['Drug'].value_counts()
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)
plt.show()

# Split the dataset into features (X) and target (y)
y = my_data['Drug']
X = my_data.drop(['Drug', 'Drug_num'], axis=1)

# Split the data into training and testing sets
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# Train a Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4, ccp_alpha=0.0, random_state=32)
drugTree.fit(X_trainset, y_trainset)

# Make predictions and evaluate accuracy
tree_predictions = drugTree.predict(X_testset)
print("Decision Tree's Accuracy (Depth=4):", metrics.accuracy_score(y_testset, tree_predictions))

# Visualize the Decision Tree
print("\nVisualizing Decision Tree...")
plot_tree(drugTree, filled=True, feature_names=X.columns, class_names=label_encoder.classes_)
plt.show()

# Train a shallower Decision Tree Classifier
print("\nTraining Shallower Decision Tree Classifier (Depth=3)...")
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=3, ccp_alpha=0.0, random_state=32)
drugTree.fit(X_trainset, y_trainset)

# Make predictions and evaluate accuracy
tree_predictions = drugTree.predict(X_testset)
print("Decision Tree's Accuracy (Depth=3):", metrics.accuracy_score(y_testset, tree_predictions))