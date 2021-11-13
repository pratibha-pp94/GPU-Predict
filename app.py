
from flask import Flask, render_template, request
import numpy as np
from predict_gpus import predict_gpus

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	
	if request.form.values():
		int_features = [int(x) for x in request.form.values()]
		year = np.array(int_features)

	prediction = predict_gpus.predict(year)
	print(prediction)
	return render_template('index.html', prediction_text="Predicted mean size of GPU memory is {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)