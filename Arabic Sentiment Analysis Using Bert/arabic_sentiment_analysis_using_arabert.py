# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 03:45:28 2020

@author: rehab
"""

import ktrain
from ktrain import text

from flask  import Flask ,jsonify ,request

#test_text = "سئ جدا هدا الفيلم "

loaded_model = ktrain.load_predictor('trained_model')

#sentiment=loaded_model.predict(test_text)

#print(sentiment)

app=Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    sentiment=loaded_model.predict(data['text'])
    
    return jsonify({'Prediction':sentiment })


if __name__ == '__main__':
    app.run(debug=False)