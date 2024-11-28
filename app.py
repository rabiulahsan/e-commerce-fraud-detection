from flask import Flask,request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd 
import math
import os
from src.exception import CustomException
from src.logger import logging
from src.pipelines.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application
CORS(app)



# Route for a home page
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Loan Eligibility Prediction API!"})

# Route for making predictions
@app.route('/predictdata', methods=['POST'])
def predictdata():
    try:
        #log the route hit
        print("hitting the route successfully...")  
        # Retrieve data from the request JSON
        data = request.json
        
        
        # Create a CustomData instance with the input data
        custom_data = CustomData(
            Multiple_Transactions=int(data.get('multiple_transactions')),
            Mismatch_Between_IP_And_Location=int(data.get('mismatch_between_ip_and_location')),
            Customer_Location=data.get('customer_location'),
            Customer_Tier=data.get('customer_tier'),
            Payment_Method=data.get('payment_method'),
            Transaction_Status=data.get('transaction_status'),
            Product_Category=data.get('product_category'),
            Transaction_Amount=float(data.get('transaction_amount')),
            Purchase_History=int(data.get('purchase_history')),
            High_Risk_Countries=int(data.get('high_risk_countries')),
            Customer_Age=int(data.get('customer_age'))
        )


        
        # Convert the input data to a DataFrame
        pred_df = custom_data.make_data_frame()
        
        # Initialize the prediction pipeline and make a prediction
        predict_pipeline = PredictPipeline()
        # Make predictions
        predictions = predict_pipeline.predict_fraud(pred_df)

        if(predictions[0] ==0):
            return jsonify({"result":"Not Fraud"})
        else:
            return jsonify({"result":"Fraud"})
        

    except Exception as e:
        raise CustomException(e)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)





    # body should like this 
