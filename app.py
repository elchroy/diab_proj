from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from numpy import loadtxt
import network

"""
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
"""

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

net = network.Network([7, 3000, 30, 1])
weights = []
biases = []

for l in xrange(net.size):
	biases.append(loadtxt("./params/biases/bias_{0}.txt".format(l)))
	weights.append(loadtxt("./params/weights/weight_{0}.txt".format(l)))

net.biases = biases
net.weights = weights

def prepare_input(request_json):
	return [
		float(request_json['no_preg']),
		float(request_json['glucose_conc']),
		float(request_json['blood_pressure']),
		float(request_json['fold_thickness']),
		float(request_json['serum_insulin']),
		float(request_json['body_mass_index']),
		float(request_json['pedigree_func']),
		float(request_json['age'])
	]

@app.route('/', methods=['GET'])
@cross_origin()
def home():
	return jsonify("Welcome to Roy's Diabetes Classifier")


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
	input_vector = prepare_input(request.json)
	prediction = net.predict(input_vector)
	message = "DIABETIC" if prediction >= 0.5 else "NOT_DIABETIC"
	res = {
		'message': message,
		'prediction': round(prediction)
	}
	return jsonify(res)


if __name__ == "__main__":
	app.run()