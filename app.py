import pickle
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from flask import Flask, request

# Load the pickle file 
with open('cancer.pkl', 'rb') as canser_model_pkl:
    model = pickle.load(canser_model_pkl)

# Initialise a Flask object
app = Flask(__name__)

# Create an API endpoint for predicting
@app.route('/predict_cancer')
def predict_cancer():

    # Read all necessary request parameters
    s1 = request.args.get('s1')
    s2 = request.args.get('s2')
    s3 = request.args.get('s3')
    s4 = request.args.get('s4')
    s5 = request.args.get('s5')
    s6 = request.args.get('s6')
    s7 = request.args.get('s7')
    s8 = request.args.get('s8')
    s9 = request.args.get('s9')
    s10 = request.args.get('s10')
    s11 = request.args.get('s11')
    s12 = request.args.get('s12')
    s13 = request.args.get('s13')
    s14 = request.args.get('s14')
    s15 = request.args.get('s15')
    s16 = request.args.get('s16')
    s17 = request.args.get('s17')
    s18 = request.args.get('s18')
    s19 = request.args.get('s19')
    s20 = request.args.get('s20')
    s21 = request.args.get('s21')
    s22 = request.args.get('s22')
    s23 = request.args.get('s23')
    s24 = request.args.get('s24')
    s25 = request.args.get('s25')
    s26 = request.args.get('s26')
    s27 = request.args.get('s27')
    s28 = request.args.get('s28')
    s29 = request.args.get('s29')
    s30 = request.args.get('s30')

    # Use the predict method of the model to get the prediction for unseen data
    data = np.array([[s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20,
                             s21, s22, s23, s24, s25, s26, s27, s28, s29, s30]]).astype(np.float64)

    result = model.predict(data)

    # return the result back
    return 'Predicted result for observation ' + str(data) + ' is: ' + str(result)

# Specify Host and port on app.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)










## http://127.0.0.1:5000/predict_cancer?s1=-0.96666522&s2=0.32786912&s3=-0.93579507&s4=-0.91104225&s5=0.60962671&s6=0.36569592&s7=-0.10914833&s8=-0.62181482&s9=-0.63860111&s10=0.53651178&s11=-0.46379509&s12=0.5132434&s13=-0.45632075&s14=-0.59189989&s15=0.67370318&s16=1.26928541&s17=2.17185315&s18=1.12535098&s19=0.64821758&s20=1.09244461&s21=-0.96440581&s22=-0.08750638&s23=-0.94145109&s24=-0.84547739&s25=-0.07511418&s26=-0.01862761&s27=-0.10400188&s28=-0.47718048&s29=-0.5634723&s30=0.05526303
