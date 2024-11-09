from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    
    with open('model_scaler.pkl', 'rb') as file:
        knn = pickle.load(file)
except (FileNotFoundError, pickle.UnpicklingError) as e:
    print(f"Error loading pickle file: {e}")
    knn = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if knn is None:
        return jsonify({'error': 'Model not loaded properly'}), 500

    
    data = request.get_json()
    total_parts_made = float(data.get('total_parts_made', 0))
    defective_parts_repairable = float(data.get('defective_parts_repairable', 0))
    defective_parts_scrap = float(data.get('defective_parts_scrap', 0))
    experience_company = float(data.get('experience_company', 0))
    leadership_experience = float(data.get('leadership_experience', 0))
    attendance_percentage = float(data.get('attendance_percentage', 0))
    late_percentage = float(data.get('late_percentage', 0))
    work_dedication_percentage = float(data.get('work_dedication_percentage', 0))
    salary = float(data.get('salary', 0))

    if any(v is None for v in [total_parts_made, defective_parts_repairable, defective_parts_scrap,
                               experience_company, leadership_experience, attendance_percentage,
                               late_percentage, work_dedication_percentage, salary]):
        return jsonify({'error': 'All fields are required.'}), 400

    
    new_employee_data = np.array([[total_parts_made, defective_parts_repairable, defective_parts_scrap,
                                   experience_company, leadership_experience, attendance_percentage,
                                   late_percentage, work_dedication_percentage]])

    
    try:
        predicted_result = knn.predict(new_employee_data)[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
    response = {'result': predicted_result}
    if predicted_result == 'Bonus':
        calculated_bonus = min(0.12 * salary, 0.08 * salary + (0.004 * total_parts_made) - (0.003 * defective_parts_scrap))
        response['calculated_bonus'] = round(calculated_bonus, 2)
    elif predicted_result == 'Increment':
        calculated_increment = min(0.08 * salary, 0.05 * salary + (0.003 * total_parts_made) - (0.002 * defective_parts_scrap))
        response['calculated_increment'] = round(calculated_increment, 2)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
