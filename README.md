This project includes:

Data loading and cleaning

Categorical encoding

Feature–target separation

Model training using Decision Tree

Evaluation (accuracy, classification report, confusion matrix)

📁 Dataset

👉 Dataset Source:
https://github.com/Prodigy-Infotech/data-science-datasets/tree/main/Task%203

File used: bank.csv

The dataset contains customer details like:

Age

Job

Marital status

Education

Loan status

Number of contacts

Outcome of previous marketing campaigns

Final subscription result (target variable: y)

🧠 Machine Learning Problem

This is a binary classification task.

Target Variable:

y = yes → Customer subscribed

y = no → Customer did not subscribe

Goal: Predict whether a new customer will subscribe to a term deposit.

🚀 Steps Performed
1. Load and Explore Dataset

Read the CSV file

View dataset shape

Display first few rows

2. Data Preprocessing

Many columns are categorical → use Label Encoding

No missing values handling required in this dataset

3. Split Data

Training set: 75%

Testing set: 25%

4. Model Training

We use:

DecisionTreeClassifier(criterion="entropy", max_depth=5)

5. Model Evaluation

We compute:

Accuracy score

Classification report (precision, recall, f1-score)

Confusion matrix
