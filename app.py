from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))
features_to_use = pickle.load(open("features.pkl", "rb"))

# Feature engineering function
def feature_engineering(df, training_columns):
    # Ratios and interaction features
    df['debt_to_income'] = df['loan_amount'] / df['income_annum']
    df['assets_to_loan'] = (df['residential_assets_value'] + df['commercial_assets_value']) / df['loan_amount']
    df['credit_income_ratio'] = df['cibil_score'] * df['income_annum']

    # Binning
    df['loan_term_bin'] = pd.cut(df['loan_term'], bins=[0,120,240,360], labels=['short','medium','long'])
    df['credit_score_bin'] = pd.cut(df['cibil_score'], bins=[300,600,700,850], labels=['low','medium','high'])

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['education','self_employed','loan_term_bin','credit_score_bin'], drop_first=True)

    # Add missing columns based on training model
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[training_columns]
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # User input
            data = {
                "income_annum": float(request.form["income_annum"]),
                "loan_amount": float(request.form["loan_amount"]),
                "loan_term": float(request.form["loan_term"]),
                "cibil_score": float(request.form["cibil_score"]),
                "no_of_dependents": int(request.form["no_of_dependents"]),
                "self_employed": request.form["self_employed"],
                "education": request.form["education"],
                "residential_assets_value": float(request.form.get("residential_assets_value", 0)),
                "commercial_assets_value": float(request.form.get("commercial_assets_value", 0))
            }

            input_df = pd.DataFrame([data])

            # Get training columns from model's training data
            if hasattr(model, "coef_"):
                # Logistic Regression: use model.coef_ to get number of features
                # We'll assume the order matches training columns
                # In practice, safest is to use a column list from training
                training_columns = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else input_df.columns
            else:
                training_columns = input_df.columns

            input_df = feature_engineering(input_df, training_columns)
            pred = model.predict(input_df)[0]
            prediction = "Approved" if pred == 1 else "Rejected"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)