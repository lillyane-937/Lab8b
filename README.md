🏡 Airbnb Listings Classification with Decision Trees

This project applies **Decision Tree Classification** to predict a specific label in the Airbnb listings dataset. Using **Python** in **Jupyter Notebook**, along with libraries like **NumPy**, **Pandas**, and **scikit-learn**, this project covers the end-to-end evaluation phase of the machine learning lifecycle.

## Project Overview

The goal of this project is to solve a classification problem by predicting a label for Airbnb listings based on chosen features. I implemented a decision tree model, fine-tuned it through **grid search** to find optimal hyperparameters, evaluated the model with multiple metrics, and saved it for future use.

### Key Tasks

1. **Data Preparation**:
   - **Load the Dataset**: Loaded the Airbnb listings data into a Pandas DataFrame.
   - **Define the Label and Features**: Specified the target variable (label) for classification and the feature set.
   - **Create Labeled Examples**: Preprocessed the data and created labeled examples to train the model.

2. **Data Splitting**:
   - Split the data into **training** and **testing** sets to evaluate model performance effectively.

3. **Model Training and Testing**:
   - **Initial Decision Tree Model**: Trained a decision tree classifier using scikit-learn’s default parameters.
   - **Hyperparameter Tuning (Grid Search)**: Conducted a grid search to identify the best hyperparameters for the decision tree, such as maximum depth and minimum samples per split, to optimize model performance.
   - **Evaluation of Optimized Model**: Re-trained, tested, and evaluated the decision tree model using the optimal hyperparameters.

4. **Model Evaluation**:
   - **Precision-Recall Curve**: Plotted precision-recall curves for both the default and optimized models to compare precision and recall at different thresholds.
   - **ROC Curve and AUC**: Generated the Receiver Operating Characteristic (ROC) curve and calculated the Area Under the Curve (AUC) for both models, helping evaluate model effectiveness.
   - **Feature Importance**: Visualized the importance of each feature in the decision tree, highlighting the most impactful predictors.

5. **Model Persistence**:
   - Saved the final model, making it reusable for future predictions without retraining.

## Tech Stack

- **Programming Language**: Python
- **Development Environment**: Jupyter Notebook
- **Libraries**:
  - **NumPy**: For data handling and mathematical operations.
  - **Pandas**: For data loading, cleaning, and manipulation.
  - **scikit-learn**: For implementing the decision tree classifier, performing grid search, and evaluating the model.

## Results

- **Model Comparison**: The optimized decision tree model demonstrated better precision, recall, and AUC scores compared to the initial model.
- **Feature Importance**: The feature importance plot provided insight into the most influential features, enhancing interpretability of the model’s decisions.

## What I Learned

Through this project, I gained hands-on experience in:
- **Model Selection and Evaluation with Decision Trees**: Understanding decision tree-based classification, model tuning, and evaluation metrics.
- **Hyperparameter Tuning with Grid Search**: Using grid search to identify optimal parameters like tree depth, improving model accuracy and preventing overfitting.
- **Model Persistence**: Saving the trained model for future use, allowing for efficient and reusable solutions.
