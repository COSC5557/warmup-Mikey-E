#Author: Michael Elgin (melgin@uwyo.edu)
#2023_10_09

#File to model the wine data

#Python 3.11.2
import numpy as np #1.26.0
import pandas as pd #1.5.3
from sklearn.model_selection import train_test_split #1.3.0
from sklearn.linear_model import LinearRegression

def LR_model():
    """Linear regression model, red chosen"""
    df = pd.read_csv("data/winequality-red.csv", sep=";")

    all_data = df.to_numpy()#Just for numpy format

    #Train a linear model
    train, test = train_test_split(all_data, test_size=0.2, random_state=0)
    X_train = train[:, 0:-1] #Features
    y_train = train[:, -1] #Target
    X_test = test[:, 0:-1] #Features
    y_test = test[:, -1] #Target
    LR_model = LinearRegression()
    LR_model.fit(X_train, y_train)

    #Evaluate the model

    #First, get a baseline
    y_base = np.median(y_test)#Equivalent of random-baseline guessing for regression
    MAPE_base = np.mean(np.abs(y_test - y_base)) * 100 #Mean absolute percentage error
    print("Baseline error for regression: {0}".format(MAPE_base))

    #Second, get the trained model's performance
    y_pred = LR_model.predict(X_test)
    MAPE_pred = np.mean(np.abs(y_test - y_pred)) * 100
    print("Model's error for regression: {0}".format(MAPE_pred))

def DT_model():
    """Decision tree model for classification"""
    pass

def view_data():
    """Brief viewing of the data head"""
    df_red = pd.read_csv("data/winequality-red.csv", sep=";")
    df_white = pd.read_csv("data/winequality-white.csv", sep=";")
    print(df_white.head())
    print(df_red.head())

def main():
    #view_data()
    LR_model()

if __name__ == "__main__":
    main()