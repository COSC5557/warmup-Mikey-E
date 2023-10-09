#Author: Michael Elgin (melgin@uwyo.edu)
#2023_10_09

#File to load the wine data

#Python 3.11.2
import numpy as np #1.26.0
import pandas as pd #1.5.3

def view_data():
    """Main function will be used for brief viewing of the data"""
    df_red = pd.read_csv("data/winequality-red.csv", sep=";")
    df_white = pd.read_csv("data/winequality-white.csv", sep=";")
    print(df_white.head())
    print(df_red.head())

def main():
    view_data()

if __name__ == "__main__":
    main()