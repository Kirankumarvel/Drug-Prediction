# Drug Classification using Decision Trees 🌳💊

## Overview

This project builds a Decision Tree Classifier to predict drug types based on patient characteristics such as age, sex, blood pressure (BP), cholesterol level, and sodium-to-potassium ratio. The dataset is preprocessed, analyzed, and used to train a model that classifies patients into different drug categories.

## Dataset

The dataset used in this project is sourced from IBM Developer Skills Network:

URL: [Drug Classification Dataset](https://example.com)

### Features:

- Age
- Sex
- Blood Pressure (BP)
- Cholesterol
- Sodium-to-Potassium Ratio (Na_to_K)

**Target Variable:** Drug Type (A, B, C, X, Y)

## Project Workflow

1. **Data Preprocessing 🧹**

    - Load the dataset using Pandas.
    - Check for missing values.
    - Encode categorical variables using `LabelEncoder`.
    - Perform correlation analysis.
    - Visualize category distribution.

2. **Model Training 🏋️‍♂️**

    - Split data into training (70%) and testing (30%) sets.
    - Train a Decision Tree Classifier (`max_depth=4`).
    - Evaluate model accuracy on test data.

3. **Model Visualization 📊**

    - Plot the decision tree structure using `plot_tree()`.
    - Train a shallower Decision Tree (`max_depth=3`) to compare performance.

## Installation & Requirements 🛠️

Ensure you have Python installed along with the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage 🚀

Run the script to train and evaluate the model:
```bash
python drug_classification_decision_tree.py
```

## Expected Output 📈

- **Decision Tree's Accuracy (Depth=4):** Varies based on dataset split.
- **Decision Tree's Accuracy (Depth=3):** Slightly lower due to reduced depth.

### Visualizations:

- Bar chart showing drug category distribution.
- Decision tree plot illustrating classification criteria.

## Key Learnings 📚

- Decision Trees are useful for interpretable machine learning.
- Feature encoding is essential for categorical variables.
- Reducing `max_depth` can help prevent overfitting, improving model generalization.


## License 📜

This project is open-source and available for educational purposes.
