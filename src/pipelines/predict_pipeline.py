import pandas as pd 
import math
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass;

    def predict_fraud(self, features):
        try:
            model_path = 'models/Gradient_Boosting_model.pkl'
            preprocessor_path = 'models/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Loading model and preprocessor file...")

            # print(model)

            data_scaled = preprocessor.transform(features)
            data_scaled_df = pd.DataFrame(data_scaled, columns = preprocessor.feature_names_in_)
            # print(data_scaled)
            preds = model.predict(data_scaled_df)

            logging.info("Prediction Completed...")

            return preds

        except Exception as e:
            raise CustomException(e)


class CustomData:
    def __init__(self,
                 Multiple_Transactions: int,  # 0 / 1
                 Mismatch_Between_IP_And_Location: int,
                 Customer_Location: str,
                 Customer_Tier: str,
                 Payment_Method: str,
                 Transaction_Status: str,
                 Product_Category: str,
                 Transaction_Amount: float,
                 Purchase_History: int,
                 High_Risk_Countries: int,  # 0 / 1
                 Customer_Age: int):

        # Initialize attributes
        self.Multiple_Transactions = Multiple_Transactions
        self.Mismatch_Between_IP_And_Location = Mismatch_Between_IP_And_Location
        self.Customer_Location = Customer_Location
        self.Customer_Tier = Customer_Tier
        self.Payment_Method = Payment_Method
        self.Transaction_Status = Transaction_Status
        self.Product_Category = Product_Category
        self.Transaction_Amount = Transaction_Amount
        self.Purchase_History = Purchase_History
        self.High_Risk_Countries = High_Risk_Countries
        self.Customer_Age = Customer_Age

    def make_data_frame(self):
        try:
            # Create a dictionary with the data
            custom_data_input_dict = {
                "Multiple_Transactions": [self.Multiple_Transactions],
                "Mismatch_Between_IP_And_Location": [self.Mismatch_Between_IP_And_Location],
                "Customer_Location": [self.Customer_Location],
                "Customer_Tier": [self.Customer_Tier],
                "Payment_Method": [self.Payment_Method],
                "Transaction_Status": [self.Transaction_Status],
                "Product_Category": [self.Product_Category],
                "Transaction_Amount": [self.Transaction_Amount],
                "Purchase_History": [self.Purchase_History],
                "High_Risk_Countries": [self.High_Risk_Countries],
                "Customer_Age": [self.Customer_Age],
            }

            # Return a DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e)


# Main function for testing
if __name__ == "__main__":
    try:
        # Create a CustomData instance with sample data
        custom_data = CustomData(
            Multiple_Transactions=1,
            Mismatch_Between_IP_And_Location=0,
            Customer_Location="India",
            Customer_Tier="new",
            Payment_Method="debit card",
            Transaction_Status="successful",
            Product_Category="home",
            Transaction_Amount=250.75,
            Purchase_History=5,
            High_Risk_Countries=0,
            Customer_Age=35
        )

        # Convert the custom data to a DataFrame
        pred_df = custom_data.make_data_frame()
        # print("Generated DataFrame:")

        # Initialize the PredictPipeline
        predict_pipeline = PredictPipeline()

        # print(pred_df.columns)
        # Make predictions
        predictions = predict_pipeline.predict_fraud(pred_df)

        if(predictions[0] ==0):
            print("Not Fraud")
        else:
            print("Fraud")
    except Exception as e:
        print(f"An error occurred: {e}")
