from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np


app = Flask(__name__)

# 피클 파일에서 훈련된 모델을 로드
with open('model/RF_BODYFAT_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Main page
@app.route('/')
def index():
    return render_template('main.html')

@app.route('/writer')
def get():
    return render_template('writer.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        "density": request.form['density'],
        "age": request.form['age'],
        "weight": request.form['weight'],
        "height": request.form['height'],
        "neck": request.form['neck'],
        "chest": request.form['chest'],
        "abdomen": request.form['abdomen'],
        "hip": request.form['hip'],
        "thigh": request.form['thigh'],
        "knee": request.form['knee'],
        "ankle": request.form['ankle'],
        "forearm": request.form['forearm'],
        "wrist": request.form['wrist']
    }

    # json으로 변환
    input_json = json.dumps(input_data)
    input_data_arr = np.array([list(input_data.values())]).astype(float)
    input_data_arr = input_data_arr.reshape(1, -1)
    
    # 예측 수행
    prediction = model.predict(input_data_arr)

    #웹 페이지 리턴
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)