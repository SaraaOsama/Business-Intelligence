#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# # Data Preprocessing

# In[3]:


data = pd.read_csv('Walmart_Dataset.csv', encoding='latin1')


# In[4]:


data.head(20)


# In[5]:


data.isnull().sum()


# In[6]:


sum(data.duplicated())


# In[7]:


data.info()


# In[8]:


# Data Exploration and Cleaning
numeric_columns = data.select_dtypes(include=np.number).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())  # Replace missing numerical values with mean
for column in data.columns:
    data[column] = data[column].fillna(data[column].mode()[0])


# In[9]:


# Data Exploration and Cleaning
print(f"Data Description : {data.describe()}")


# In[10]:


# Print summary statistics and information about the dataset
print(f"Data Information : {data.info()}")


# In[11]:


sns.boxplot(data=data[['Sales']])


# In[12]:


# Handling outliers in the 'Sales' column using IQR
Q1 = data['Sales'].quantile(0.25)
Q3 = data['Sales'].quantile(0.75)
IQR = Q3 - Q1
filter = (data['Sales'] >= Q1 - 1.5 * IQR) & (data['Sales'] <= Q3 + 1.5 * IQR)
data = data.loc[filter]


# In[13]:


sns.boxplot(data=data[['Sales']])


# # Data Visualization

# In[14]:


# Data Visualization
# Visualizing sales by product category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Sales', data=data)
plt.title('Sales by Product Category')
plt.show()


# In[15]:


# Visualizing sales by Segments
plt.figure(figsize=(10, 6))
sns.barplot(x='Segment', y='Sales', data=data)
plt.title('Sales by Segment')
plt.show()


# In[16]:


# Visualizing sales by region
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Sales', data=data, estimator=np.sum)
plt.title('Total Sales by Region')
plt.show()


# In[17]:


# Visualizing sales by state
plt.figure(figsize=(12, 6))
sns.barplot(x='State', y='Sales', data=data, estimator=np.sum)
plt.title('Total Sales by State')
plt.xticks(rotation=90)
plt.show()


# In[18]:


# Visualizing sales by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='Sales', data=data, estimator=np.sum)
plt.title('Total Sales by City')
plt.xticks(rotation=90)
plt.show()


# In[19]:


# plotting the correlation matrix
data.corr()
sns.set(rc={'figure.figsize': (15,10)})
sns.heatmap(data.corr(),annot=True)


# # Linear Regression Model and its Evaluation

# In[20]:


# Machine Learning Model for Sales Prediction
# Prepare data for regression model
X = pd.get_dummies(data[['Category', 'Region', 'Quantity', 'Discount', 'Profit', 'Ship Mode']])
y = data['Sales']


# In[21]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[23]:


y_pred = model.predict(X_test)


# In[24]:


# Make predictions and evaluate the model
predictions = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R^2 Score:", r2_score(y_test, predictions))


# # Polynomial Regression Model and its Evaluation

# In[40]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=False)


# In[41]:


poly_feature=poly.fit_transform(X)


# In[42]:


x_train_poly,x_test_poly,y_train_poly,y_test_poly=train_test_split(poly_feature,y,test_size=0.4,random_state=20)


# In[43]:


reg=LinearRegression()
reg.fit(x_train_poly,y_train_poly)


# In[44]:


y_predict_poly=reg.predict(x_test_poly)
print("Mean Squared Error:", mean_squared_error(y_test_poly, y_predict_poly))
print("R^2 Score:", r2_score(y_test_poly, y_predict_poly))


# # Decision Tree Regressor Model and its Evaluation

# In[45]:


from sklearn.tree import DecisionTreeRegressor
param_grid = {'max_depth': range(1, 20), 'min_samples_split': range(2, 10)}
DTR = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5)


# In[46]:


DTR.fit(X_train, y_train)


# In[47]:


y_pred=DTR.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


# # Random Forest Regressor Model and its Evaluation

# In[48]:


from sklearn.ensemble import RandomForestRegressor
RFR=RandomForestRegressor()


# In[49]:


RFR.fit(X_train,y_train)


# In[50]:


y_pred=RFR.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))


# # Time Series

# In[51]:


# Time Series Analysis for Sales
# Convert 'Order Date' to datetime and set it as index
data['Order Date'] = pd.to_datetime(data['Order Date'])
data.set_index('Order Date', inplace=True)


# In[52]:


monthly_sales = data['Sales'].resample('MS').sum()
plt.figure(figsize=(12, 6))
monthly_sales.plot(title='Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


# In[53]:


# Check if the time series is stationary
result = adfuller(monthly_sales)
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])


# In[54]:


# If not stationary, make it stationary
monthly_sales_diff = monthly_sales.diff().dropna()


# In[55]:


# Plot ACF and PACF charts
plt.figure(figsize=(12, 6))
plot_acf(monthly_sales_diff, lags=22)
plt.show()


# In[56]:


plt.figure(figsize=(12, 6))
plot_pacf(monthly_sales_diff, lags=22)
plt.show()


# In[57]:


# Build ARIMA model
model = ARIMA(monthly_sales, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())


# In[58]:


# Predict future sales
forecast = model_fit.forecast(steps=12)
print(forecast)


# In[59]:


# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, label='Historical Sales')
plt.plot(forecast, label='Forecasted Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.show()


# In[ ]:




