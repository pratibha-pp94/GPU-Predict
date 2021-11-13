# -*- coding: utf-8 -*-
"""Using Regression to predict GPUs of the future_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v696z5ZUOgI0q5stdQmYHB039loB9_qz
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class predict_gpus:
    def predict(year):
        #Import dataset from csv file
        dataset = pd.read_csv(r'D:\QMUL\MSC_PRJ\nirbhay\MainProject\MainProject\All_GPUs.csv')

        """Let's try to summarize the Dataset"""

        main_columns = ['Best_Resolution', 'Core_Speed', 'Manufacturer', 'Memory', 'Memory_Bandwidth', 'Name', 'Release_Date']
        dataset = dataset[main_columns]

        """Data preprocessing & feature engineering"""

        dataset['Release_Date']=dataset['Release_Date'].str[1:-1]
        dataset=dataset[dataset['Release_Date'].str.len()==11]
        dataset['Release_Date']=pd.to_datetime(dataset['Release_Date'], format='%d-%b-%Y')
        dataset['Release_Year']=dataset['Release_Date'].dt.year
        dataset['Release_Month']=dataset['Release_Date'].dt.month
        dataset['Release']=dataset['Release_Year'] + dataset['Release_Month']/12

        dataset['Memory'] = dataset['Memory'].str[:-3].fillna(0).astype(int)

        # Numpy array that holds unique release year values
        year_array = dataset.sort_values("Release_Year")['Release_Year'].unique()
        # Numpy array that holds mean values of GPUs memory for each year
        mean_array_mem = dataset.groupby('Release_Year')['Memory'].mean().values
        # Numpy array that holds median values of GPUs memory for each year
        median_array_mem = dataset.groupby('Release_Year')['Memory'].median().values

        # Minimal value of release year from dataset
        min_year = year_array[0]
        # Median size of memory in min_year
        median_min_year = median_array_mem[0]

        """Creating Calculated model and fitting exponential curve"""

        # Function to calculate size of memory based on Moore's law
        def calMooresValue(x, y_trans):
            return median_array_mem[0] * 2**((x-y_trans)/2)

        # GPU Memory Size calculation based on Moore's Law
        y_pred_calMoore = calMooresValue(year_array, int(min_year))

        # Fitting exponential curve to dataset
        def exponentialCurve(x, a, b, c):
            return a*2**((x-c)*b)

        popt, pcov = curve_fit(exponentialCurve,  year_array, mean_array_mem,  p0=(2, 0.5, 1998))
        y_pred_MooreFit = exponentialCurve(year_array, *popt)

        """Polynomial regression prediction model"""

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures

        # Fitting Polynomial Regression to the dataset
        ploy_regg_2 = PolynomialFeatures(degree = 2, include_bias=False)
        ploy_regg_3 = PolynomialFeatures(degree = 3, include_bias=False)

        X_poly_2 = ploy_regg_2.fit_transform(year_array.reshape(-1, 1))
        X_poly_3 = ploy_regg_3.fit_transform(year_array.reshape(-1, 1))

        linear_reg_2 = LinearRegression()
        linear_reg_3 = LinearRegression()

        linear_reg_2.fit(X_poly_2, mean_array_mem)
        linear_reg_3.fit(X_poly_3, mean_array_mem)

        y_pred_linear_reg_2 = linear_reg_2.predict(ploy_regg_2.fit_transform(year_array.reshape(-1, 1)))
        y_pred_linear_reg_3 = linear_reg_3.predict(ploy_regg_3.fit_transform(year_array.reshape(-1, 1)))

        """Selecting best model"""

        from sklearn.metrics import r2_score

        # 2nd degree curve
        score = r2_score(y_pred_linear_reg_2, mean_array_mem)
        print("r2 of 2nd degree curve is equal " + str(round(score, 3)))
        # 3rd degree curve
        score = r2_score(y_pred_linear_reg_3, mean_array_mem)
        print("r2 of 3rd degree curve is equal " + str(round(score, 3)))
        # Calculated Moore's Law curve
        score = r2_score(y_pred_calMoore, mean_array_mem)
        print("r2 of Calculated Moore's Law curve is equal " + str(round(score, 3)))
        # Fitted Moore's Law curve
        score = r2_score(y_pred_MooreFit, mean_array_mem)
        print("r2 of Fitted Moore's Law curve is equal " + str(round(score, 3)))

        ### """ Basing on above r2 scores selecting Fitted Moore's Law curve for predicting GPUs Mean Memory Size."""

        X_grid = np.arange(min(year_array), max(year_array) + 5, 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))

        y_pred_linear_reg_2 = linear_reg_2.predict(ploy_regg_2.fit_transform(X_grid))
        y_pred_linear_reg_3 = linear_reg_3.predict(ploy_regg_3.fit_transform(X_grid))

        X_grid = X_grid.flatten()

        y_pred_calMoore = calMooresValue(X_grid, int(min_year))
        y_pred_MooreFit = exponentialCurve(X_grid, *popt)



        #"""Predicting GPUs mean memory size in 2025"""

        memory_year = exponentialCurve(year, *popt)

        print("base :", memory_year)

        print("\n Predicted mean size of GPU memory in 2025 is " + str(round(int(memory_year) / 1024, 2)) + " GB.")

        ans = str(round(int(memory_year) / 1024, 2))

        return ans