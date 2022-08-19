import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
pd.set_option('display.max_rows',10)


# Data import

data = pd.read_csv(r'xxxxxx\Forecasting Challenge Data.csv', header=None)


# Transforming data into usuable format

df=pd.DataFrame(data)
df.columns=['Date','v1','v2','v3','v4','v5','Target']
df['Date'] = df['Date'].astype('datetime64[ns]')
df['DoW']=df['Date'].dt.dayofweek
df['Month']=df['Date'].dt.month
df['Year']=df['Date'].dt.year.astype('int64')
pd.to_numeric(df['Year'])


df["logv2"] = np.log(df["v2"])
df["logv3"] = np.log(df["v3"])


# Linear Regression
mask = (df['Year'] >= 2012) & (df['Year'] <= 2017)
df1= df.loc[mask]
y=df1["Target"]
y=np.array(y)

x=df1[['v1', 'logv2', 'logv3', 'v4']]

regr = linear_model.LinearRegression()
regr.fit(x, y)
model = ols('Target ~ logv2 + C(Month) + C(DoW)', data=df1).fit()
print_model = model.summary()

print(print_model)

print(regr.coef_)
print(model.params)


# Model Evaluation Plots

model_fit = model
model_fitted_y = model_fit.fittedvalues
# model residuals
model_residuals = model_fit.resid

# normalized residuals
model_norm_residuals = model_fit.get_influence().resid_studentized_internal

# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# absolute residuals
model_abs_resid = np.abs(model_residuals)

# leverage points
model_leverage = model_fit.get_influence().hat_matrix_diag

# cook's distance
model_cooks = model_fit.get_influence().cooks_distance[0]

plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, df1.columns[-1], data=df1,
                          lowess=True,
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals');