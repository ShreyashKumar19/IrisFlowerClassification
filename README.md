# Iris Flower Classification

## About the Project

This is my first machine learning project. In this project, I have used machine learning to predict the species of an iris flower based on its measurements.

The project can predict three types of iris flowers:

* Setosa
* Versicolor
* Virginica

The prediction is based on four measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

## Dataset

I used the Iris dataset from scikit-learn.

It contains 150 samples of iris flowers. Each sample has four measurements and one species label.

## What I Did in This Project

### 1. Loaded the Dataset

I loaded the Iris dataset using scikit-learn and converted it into a pandas DataFrame.

### 2. Explored the Data

I checked the dataset to understand it better. I checked:

* First few rows
* Number of rows and columns
* Missing values
* Number of flowers in each species
* Basic statistics

### 3. Visualized the Data

I used scatter plots and pair plots to see how the different measurements are related to each other and how the three species are different.

### 4. Split the Data

I divided the data into training and testing data.

* 80% for training
* 20% for testing

### 5. Scaled the Data

I used `StandardScaler` to scale the four input features before training the model.

### 6. Trained the Model

I used **Logistic Regression** to classify the flowers.

The model was trained using the training data and then tested on the test data.

### 7. Checked the Model

I checked the model performance using:

* Accuracy
* Classification report
* Confusion matrix

The model achieved **93.33% accuracy** on my test data.

### 8. Made a Simple Web App

I saved the trained model and scaler using Joblib.

Then I created a simple Streamlit application where the user can enter the four flower measurements and get the predicted species.

## How to Run

First, install the required libraries:

pip install -r requirements.txt

Then run the training file:

python iris_classification.py

This will create these two files:

iris_model.pkl -> contains the trained Logistic Regression model
iris_scaler.pkl -> contains the scaler used to preprocess the input data

After that, start the Streamlit app:

streamlit run iris_app.py

The app will open in the browser.

The training file does not need to be run every time. Once the model files have been created, I can directly run the Streamlit app.

## Example

Input:
Sepal Length = 5.1
Sepal Width = 3.5
Petal Length = 1.4
Petal Width = 0.2

Output:
Predicted Species: Setosa

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
