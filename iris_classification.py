import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Load the Iris dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Change numbers into species names
df["species"] = df["species"].replace({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("First 5 rows:")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSpecies count:")
print(df["species"].value_counts())

print("\nBasic statistics:")
print(df.describe())


# --- Data Visualization ---

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
plt.show()


sns.pairplot(df, hue="species")
plt.show()


# --- Prepare the data ---

X = df.drop("species", axis=1)
y = df["species"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# --- Scale the data ---

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Logistic Regression ---

logistic_model = LogisticRegression()

logistic_model.fit(X_train_scaled, y_train)

prediction = logistic_model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nLogistic Regression Accuracy:")
print(accuracy)


# --- Classification report ---

print("\nClassification Report:")
print(classification_report(y_test, prediction))


# --- Confusion Matrix ---

cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Setosa", "Versicolor", "Virginica"],
    yticklabels=["Setosa", "Versicolor", "Virginica"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# --- Save the model ---

joblib.dump(logistic_model, "iris_model.pkl")
joblib.dump(scaler, "iris_scaler.pkl")

print("\nModel saved successfully.")
print("Scaler saved successfully.")


# --- Test with a new flower ---

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)

new_flower_scaled = scaler.transform(new_flower)

new_prediction = logistic_model.predict(new_flower_scaled)

print("\nPrediction for new flower:")
print(new_prediction[0])