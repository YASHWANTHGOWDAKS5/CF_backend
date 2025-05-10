from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS immediately after Flask app initialization

# Load the models
model_min = joblib.load("model_min.pkl")
model_max = joblib.load("model_max.pkl")

# Load dataset to get mode values
data = pd.read_csv("35985678-0d79-46b4-9ed6-6f13308a1d24_c5de158f10a66787f829ed3674659014.csv")
data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
data = data.dropna(subset=["Variety", "Grade"])
mode_variety = data["Variety"].mode()[0]
mode_grade = data["Grade"].mode()[0]

@app.route("/predict", methods=["POST"])
def predict_price():
    req_data = request.get_json()

    # Check if the required keys are in the incoming JSON
    if "date" not in req_data or "market" not in req_data:
        return jsonify({"error": "Missing required data: 'date' and 'market' are required."}), 400

    try:
        date_str = req_data["date"]
        market = req_data["market"]

        # Convert the date string to a datetime object
        user_date = datetime.strptime(date_str, "%d-%m-%Y")
        month = user_date.month
        year = user_date.year

        # Prepare the user input dataframe
        user_input = pd.DataFrame([{
            "Month": month,
            "Year": year,
            "Market": market,
            "Variety": mode_variety,
            "Grade": mode_grade
        }])

        # Predict min and max prices
        min_price = model_min.predict(user_input)[0]
        max_price = model_max.predict(user_input)[0]

        # Return the response
        return jsonify({
            "min_price": float(round(min_price, 2)),
            "max_price": float(round(max_price, 2))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
