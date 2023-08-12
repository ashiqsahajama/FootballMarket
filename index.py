from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)

def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 11)
	loaded_model = pickle.load(open("Market_value.pkl", "rb"))
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/', methods = ['POST'])
def result():
	if request.method == 'POST':
		arr = []
		name = request.form.get("pname")
		arr.append(int(request.form.get("age")))
		arr.append(int(request.form.get("league")))
		arr.append(int(request.form.get("pos")))
		arr.append(int(request.form.get("mp")))
		arr.append(int(request.form.get("ms")))
		arr.append(int(request.form.get("ga")))
		arr.append(int(request.form.get("pk")))
		arr.append(int(request.form.get("fk")))
		arr.append(int(request.form.get("tc")))
		arr.append(int(request.form.get("tcw")))
		arr.append(int(request.form.get("pass")))
		prediction = ValuePredictor(arr)
		return render_template("result.html", prediction=[name, round(prediction[0],2)])


if(__name__ == '__main__'):
	app.run()
