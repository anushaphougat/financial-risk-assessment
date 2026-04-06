import pandas as pd
import sklearn.model_selection 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,classification_report
import pickle

# Load dataset
data = pd.read_csv(r"c:\Users\anusha\OneDrive\Documents\Downloads\archive\dataset.csv")
data.columns = data.columns.str.strip()  # Remove any leading/trailing whitespace from column names

# Select useful columns
df = data[["income_annum", "loan_amount", "loan_term", "cibil_score", "no_of_dependents", "self_employed", "education", "loan_status"]]

# Handle missing values
df = df.dropna()

# Convert categorical variables to numeric
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0}).fillna(0)
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0}).fillna(0)

# Convert target
df["loan_status"] = df["loan_status"].str.strip().map({"Approved": 1, "Rejected": 0})

# Feature engineering
df["debt_income_ratio"] = df["loan_amount"] / df["income_annum"]
df["loan_term_years"] = df["loan_term"] / 12
df["cibil_income_interaction"] = df["cibil_score"] * df["income_annum"]

# Split data into features and target
X = df[["income_annum", "loan_amount", "loan_term", "cibil_score", "no_of_dependents", "self_employed", "education", "debt_income_ratio", "loan_term_years", "cibil_income_interaction"]]
y = df["loan_status"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train) 

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
with open("model.pkl", "wb") as f:
	pickle.dump(model, f)

features_to_use = X.columns.tolist()
with open("features.pkl", "wb") as f:
	pickle.dump(features_to_use, f)

print("Model trained successfully!")

