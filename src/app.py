from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    print(data)

    age = int(data['age'])
    gender = int(data['gender'])
    height_feet = float(data['height_feet'])
    height_inches = float(data['height_inches'])
    weight_pounds = float(data['weight_pounds'])
    avg_glucose_level = float(data['avg_blood_sugar'])
    hypertension = int(data['hypertension'])
    heart_disease = int(data['heart_disease'])
    smoking_status = int(data['smoking_status'])
    ever_married = int(data['ever_married'])
    work_type_private = int(data['work_type_private'])
    work_type_govt_job = int(data['work_type_govt_job'])
    work_type_self_employed = int(data['work_type_self_employed'])
    work_type_never_worked = int(data['work_type_never_worked'])

    # Convert height from feet and inches to total inches
    total_height_inches = (height_feet * 12) + height_inches

    # Calculate BMI
    bmi = (weight_pounds / (total_height_inches ** 2)) * 703
    print(bmi)

    if (bmi >= 18.5 and bmi <= 25):
        bmi = 1
    else:
        bmi = 0

    # Load scaler and model
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    # Make predictions for the next 15 years
    probabilities = []
    age_list = []
    input_dict = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status],
        'work_type_govt_job': [work_type_govt_job],
        'work_type_never_worked': [work_type_never_worked],
        'work_type_private': [work_type_private],
        'work_type_self_employed': [work_type_self_employed]
    }

    # Create a DataFrame from the dictionary
    input_data = pd.DataFrame(input_dict)

    # Preprocess input data using the scaler
    input_data_preprocessed = scaler.transform(input_data)
    probability = model.predict_proba(input_data_preprocessed)[:, 1]
    probabilities.append(probability[0])
    age_list.append(age)

    for i in range(15):
        age += 1
        input_data.loc[0, 'age'] = age
        input_data_preprocessed = scaler.transform(input_data)
        probability = model.predict_proba(input_data_preprocessed)[:, 1]
        probabilities.append(probability[0])
        age_list.append(age)

    return jsonify(
        {
            'age': age_list,
            'probability': probabilities
        }
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002)
