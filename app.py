from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import psycopg2

app = Flask(__name__)
file_name = 'logistic_model.pkl'
model = joblib.load(file_name)

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    ram = float(data.get('ram'))
    battery_power = float(data.get('battery_power'))
    px_width = float(data.get('px_width'))
    px_height = float(data.get('px_height'))

    features = [ram,battery_power, px_width,px_height]
    if any(features)is None:
        return jsonify({"error": "Missing one or more input"}), 400
    
    
    prediction = model.predict([features]).tolist()[0]


    conn = psycopg2.connect(
        dbname="predicted_values",
        user="cesia",
        password="pass1",
        host="localhost",
        port="5432"
    )


    cur = conn.cursor()

   
    cur.execute(
        "INSERT INTO phone_data (ram, batter_power, px_width, px_height) VALUES (%s, %s, %s,%s)",
        (ram,battery_power, px_width,px_height)
    )

    
    conn.commit()

   
    cur.close()
    conn.close()
    return jsonify({'prediction': round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)

