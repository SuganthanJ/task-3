import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
#  Load the Bank Marketing Dataset
# -------------------------------
df = pd.read_csv("bank.csv")   # Update file name if needed

print("\n------ FIRST 5 ROWS ------")
print(df.head())

print("\n------ SHAPE OF DATASET ------")
print(df.shape)

print("\n------ COLUMN NAMES ------")
print(df.columns)

# -------------------------------
#  Encode Categorical Columns
# -------------------------------
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = label_encoder.fit_transform(df[col])

print("\n------ DATA AFTER ENCODING ------")
print(df.head())

# -------------------------------
#  Define Features (X) and Target (y)
# -------------------------------
X = df.drop("y", axis=1)   # Target: y (yes/no)
y = df["y"]

# -------------------------------
#  Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -------------------------------
#  Train Decision Tree Classifier
# -------------------------------
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)
model.fit(X_train, y_train)

# -------------------------------
#  Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
#  Model Evaluation
# -------------------------------
print("\n------ ACCURACY ------")
print(accuracy_score(y_test, y_pred))

print("\n------ CLASSIFICATION REPORT ------")
print(classification_report(y_test, y_pred))

print("\n------ CONFUSION MATRIX ------")
print(confusion_matrix(y_test, y_pred))
