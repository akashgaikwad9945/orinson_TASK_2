# Simple Linear Regression Model

This project implements a simple Linear Regression model to predict a numerical target variable based on a single feature. The project includes dataset creation, model training, evaluation, and visualization of the results.

## Objective

- Build a Linear Regression model to predict a numerical value using one feature.
- Train and evaluate the model on a small dataset.
- Visualize the regression line over the data points.

---

## Features

1. **Dataset Creation**: 
   - A synthetic dataset is generated with one feature and one target variable following a linear relationship with added noise.

2. **Model Training**:
   - The dataset is split into training and testing sets.
   - A Linear Regression model is trained on the training data.

3. **Model Evaluation**:
   - Evaluate the model using performance metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R² Score

4. **Visualization**:
   - A scatter plot of the dataset with the regression line overlaid is created.

---

## Technologies Used

- **Python**: Programming language for implementation.
- **NumPy**: For numerical operations and dataset generation.
- **Pandas**: For data manipulation (optional).
- **Scikit-learn**: For building and evaluating the Linear Regression model.
- **Matplotlib**: For plotting data and the regression line.

---

## How It Works

1. **Dataset Generation**:
   - A synthetic dataset is created with a single feature and a target variable (`y`) that has a linear relationship with the feature (`X`).
   
2. **Model Training**:
   - The data is split into training and testing sets (80%-20%).
   - A Linear Regression model is trained on the training set using Scikit-learn.

3. **Model Evaluation**:
   - The model's predictions on the test set are evaluated using:
     - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors.
     - **Mean Squared Error (MSE)**: Measures the average squared errors.
     - **R² Score**: Indicates how well the model fits the data.

4. **Visualization**:
   - The scatter plot of the dataset is created, and the regression line is drawn to show the model's predictions.

---

## Installation

1. Clone the repository or download the script:
   ```bash
   git clone https://github.com/yourusername/simple-linear-regression.git
   cd simple-linear-regression
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

3. Run the script:
   ```bash
   python linear_regression.py
   ```

---

## Usage

- Modify the dataset generation section if you'd like to use custom data.
- Run the script to train the model, evaluate its performance, and visualize the results.
- Adjust the train-test split ratio, noise level, or other parameters to experiment with the model.

---

## Results

1. **Performance Metrics**:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score

2. **Scatter Plot**:
   - Displays the data points and the fitted regression line.

---

## Example Output

### Performance Metrics:
```
Mean Absolute Error (MAE): 1.25
Mean Squared Error (MSE): 2.75
R² Score: 0.95
```

### Visualization:
- A scatter plot with the regression line overlaid.

---



## Acknowledgements

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

Feel free to reach out for any clarifications or enhancements! 😊

---
