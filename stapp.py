# importing required packages for creating the app

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np


# importing data analysis elements

from math import sqrt,ceil
import matplotlib.pyplot as plt
import time
import requests
import json


# importing machine learning libraries

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

# Title and subtitle of our Streamlit app

st.title('Predict price of your future house ')

# Progress bar

def status():
	latest_iteration = st.empty()
	bar = st.progress(0)
	for i in range(100):
		latest_iteration.text(f'Finding a property for you : {i+1} %')
		bar.progress(i + 1)
		time.sleep(0.05)
		st.empty()

st.subheader('Predicting price using a pipeline of machine learning regressors with a 10 % leeway for more choices  ')

# Writing functions for each task

@st.cache
def dataloader():

# Loading the excel file and cleaning it to remove invalid fields like negative prices or numbers

	df=pd.read_excel('data.xls')
	df=df.drop(['country'],axis=1)
	df=df[df['price']>0]
	df.rename(columns={'statezip':'zip'}, inplace=True)
	df['zip']=df['zip'].str.replace('WA','').astype(int)
	df['floors']=df['floors'].astype(int)
	df=df[df['bedrooms']>0]
	df=df[df['bathrooms']>0]
	return df
	
# returns the clean dataset for calculations and model fitting

df=dataloader()

# attributes of a house, note that 1.5 bedroom means 1 standard size and 1 smal size bedroom. 1.5 eliminated the need to write two columns

st.sidebar.subheader('Property Options')


# Sidebar Options
st.sidebar.text("Keep low space(sq.ft) requirement for more choices")
#Accepting values from the user
params={
'bedrooms' : st.sidebar.selectbox('Bedrooms needed',(1,2,3,4,5)),
'bathrooms' : st.sidebar.selectbox('Bathrooms needed(1 small=0.5,1 big=1)',(1,1.5,2,2.5,3,3.5,4)),
'floors' : st.sidebar.selectbox('Number of floors available',(df['floors'].unique())),
'sqft' : st.sidebar.slider('Minimum Space required(in Sq.ft)', 800,max(df['sqft_living']),step=100),
'waterfront':1 if st.sidebar.checkbox('I want a water facing property') else 0
}
st.sidebar.text(" Note that there are only a few water facing properties")
# This file contains the zipcodes, lat and long values ( API's are not free! )

locate=pd.read_excel('USAcoords.xlsx')


# Function to extract lat lon with zipcode as input 
# Finding and mapping the lat and lon coordinates for the map display of properties
@st.cache
def get_locations(zipcode):
	lat=float(locate.loc[locate['ZIP']==zipcode,'LAT'].iloc[0])
	lon=float(locate.loc[locate['ZIP']==zipcode,'LNG'].iloc[0])
	return lat,lon

# returning a dataframe with user accepted parameters merged with location coordinates

def map_df(df):
	df=df[df['bedrooms']==params['bedrooms']]
	df=df[df['bathrooms']==params['bathrooms']]
	df=df[df['floors']==params['floors']]
	df=df[df['waterfront']==params['waterfront']]
	df=df[(df['sqft_living']>0.9*params['sqft']) & (df['sqft_living']<1.1*params['sqft'])]
	df.reset_index()

# mapping zipcodes to coordinates

	df['lat']=[get_locations(i)[0]  for i in list(df['zip'])]
	df['lon']=[get_locations(i)[1]  for i in list(df['zip'])]
	return df


# Pipeline for predicting price using Regression models

@st.cache
def MLregressors():
	y=df['price']
	X=df[['bedrooms','bathrooms','floors','sqft_living','waterfront']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
	models = [RandomForestRegressor(n_estimators=300,max_depth=25),DecisionTreeRegressor(max_depth=25),GradientBoostingRegressor(learning_rate=0.01,n_estimators=300,max_depth=25), LinearRegression(n_jobs=10, normalize=True)]
	df_models = pd.DataFrame()
	temp = {}
	print(X_test)

#running through models

	for model in models:
		print(model)
		m = str(model)
		temp['Model'] = m[:m.index('(')]
		model.fit(X_train, y_train)
		temp['RMSE_Price'] = sqrt(mse(y_test, model.predict(X_test)))
		temp['Pred Value']=model.predict(pd.DataFrame(params,  index=[0]))[0]
		print('RMSE score',temp['RMSE_Price'])
	
	# this dataframe contains RMSE scores and Prediction value(price)

		df_models = df_models.append([temp])
		
	df_models.set_index('Model', inplace=True)
	pred_value=df_models['Pred Value'].iloc[[df_models['RMSE_Price'].argmin()]].values.astype(float)
	return pred_value, df_models

# Driver function which drives the entire inputs,model to output maps.
def run():
# Progress bar
	status()

	# Obtaining predictions and RMSE scores

	df_models=MLregressors()[0][0]
	st.write('As per your choices, the predicted price is  $ {} '.format(ceil(df_models)))
	df1=map_df(df)

	# Resetting dataframe

	dataloader()
	st.subheader("{} properties were found".format(len(df1)))

	# producing the map with locations of properties highlighted in it

	st.map(df1)
	df1.reset_index(drop=True)
	house = df1['street']
	city=df1['city']

	# displaying dataframe which contains details of found out properties

	df1.drop(labels=['street','date','lat','lon','view','waterfront','city'], axis=1,inplace = True)
	df1.insert(0, 'street', house)
	df1.insert(1, 'city', city)
	df2=df1.reset_index(drop=True)
	df2
	
# function to show model score
	
def show_ML():
	df_models=MLregressors()[1]
	df_models
	st.write('** Root mean sq error for all models**')
	st.bar_chart(df_models['RMSE_Price'])

# Predict button trigger

btn = st.sidebar.button("Predict")
if btn:
	run()
else:
	pass

st.sidebar.subheader('Model information')

if st.sidebar.checkbox('Show '):
	run()
	df_models=MLregressors()[1]
	#df_models
	st.write('**Root mean square error chart**')
	st.bar_chart(df_models['RMSE_Price'])

