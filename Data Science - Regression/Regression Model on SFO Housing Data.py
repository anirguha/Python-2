#%% md
# ## Regression Project
# 
# Build a linear regression model that predicts the `price` column in the dataset on San Francisco Apartment rentals. Make sure to go through all the the relevant steps of the modelling workflow.
# 
# 1. Data Cleaning has already been performed on this data
# 2. Explore the data, keeping an eye on useful features and potential issues 
# 3. Choose a model validation scheme (simple or cross validation)
# 4. Fit a Linear Regression Model
# 5. Evaluate your model - check assumptions, metrics, and coefficient estimates
# 6. Engineer Features as needed
# 7. Repeat Steps 2, 4, 5, 6 as needed
# 8. Once you are ready, fit your final model and report final model performance estimate by scoring on the test data. Report both test R-squared and MAE.
# 
# A reasonable goal is to get a validation MAE of <= 500 dollars.
# 
# Advice:
# 
# 1. "Perfect" is the enemy of "Good".
# 2. You will not get to an MAE of 0, we don't have perfect data, and there is some randomness and emotion in how things like apartment prices are set. 
# 3. Modelling is challenging, especially if you're new. There isn't a "right" answer, but some models are better than others. Remember - "All Models are wrong, but some are useful" 
# 4. We will likely end up with different models and performance metrics, and that's ok. You can always implement new ideas after watching the solution video. I didn't do anything too fancy, so you might think of some great ideas I didn't!
# 5. Enjoy the process, and remember that at some point it's time to let the model go. No amount of feature engineering will achieve a perfect model.
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse

rentals_df = pd.read_csv('/Users/AnirbanGuha/Library/CloudStorage/OneDrive-Personal/Maven Analytics Courses/Data Science in Python - Regression/Course Materials/Data/sf_clean.csv')

rentals_df.head()
#%% md
# #### 1. EDA
#%% md
# ### Data Dictionary
# 
# 1. Price: The price of the rental and our target variable
# 2. sqft: The area in square feet of the rental
# 3. beds: The number of bedrooms in the rental
# 4. bath: The number of bathrooms in the rental
# 5. laundry: Does the rental have a laundry machine inside the house, a shared laundry machine, or no laundry on site?
# 6. pets: Does the rental allow pets? Cats only, dogs only or both cats and dogs?
# 7. Housing type: Is the rental in a multi-unit building, a building with two units, or a stand alone house? 
# 8. Parking: Does the apartment off a parking space? No, protected in a garage, off-street in a parking lot, or valet service?
# 9. Hood district: Which part of San Francisco is the apartment located?
#%%
rentals_df.info()
#%% md
# ## EDA
# 
# 1. Based on the range of prices below, we may need to subset our data based on some value to predict more "realistic" apartments. Possibly subset based on square-footage.
# 
# 2. The 'hood_district' feature was read in as an integer but is really a categorical feature. Let's fix that.
# 
#%%
rentals_df["hood_district"] = rentals_df["hood_district"].astype("object") 
#%%
rentals_df.describe()
#%%
sns.histplot(rentals_df, x="price");
#%%
sns.heatmap(
    rentals_df.corr(numeric_only=True), 
    vmin=-1, 
    vmax=1, 
    cmap="coolwarm",
    annot=True
);
#%%
sns.pairplot(rentals_df,corner=True);
#%% md
# # Feature Engineering
# 
# 1. Group Categories together
# 2. Trying a Squared Term for Bedrooms, sqft, and bath
#%%
laundry_map = {
    "(a) in-unit": "in_unit",
    "(b) on-site": "not_in_unit",
    "(c) no laundry": "not_in_unit",
}

pet_map = {
    "(a) both": "allows_dogs",
    "(b) dogs": "allows_dogs",
    "(c) cats": "no_dogs",
    "(d) no pets": "no_dogs"
}


housing_type_map = {
    "(a) single": "single",
    "(b) double": "multi",
    "(c) multi": "multi",
}

district_map = {
    1.0: "west",
    2.0: "southwest",
    3.0: "southwest",
    4.0: "central",
    5.0: "central",
    6.0: "central",
    7.0: "marina",
    8.0: "north beach",
    9.0: "FiDi/SOMA",
    10.0: "southwest"
    
}
#%%
eng_df = rentals_df.assign(
#     hood_district = rentals_df["hood_district"].map(district_map),
#     housing_type = rentals_df["housing_type"].map(housing_type_map),
#     pets = rentals_df["pets"].map(pet_map),
#     laundry = rentals_df["laundry"].map(laundry_map),
    sqft2 = rentals_df["sqft"] ** 2,
    sqft3 = rentals_df["sqft"] ** 3,
    beds2 = rentals_df["beds"] ** 2,
    beds3 = rentals_df["beds"] ** 3,
    bath2 = rentals_df["bath"] ** 2,
    bath3 = rentals_df["bath"] ** 3,
    beds_bath_ratio = rentals_df["beds"] / rentals_df["bath"]
)

eng_df = pd.get_dummies(eng_df, drop_first=True, dtype="int")
#%%
eng_df.head()
#%% md
# #### 2. Modeling
#%%
from sklearn.model_selection import train_test_split

target = "price"
drop_cols = [
#     "pets_no_dogs",
#     "housing_type_single"
]

X = sm.add_constant(eng_df.drop([target] + drop_cols, axis=1))

# Log transform slightly improves normality
y = np.log(eng_df[target])
# y = eng_df[target]

# Test Split
X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=2023)
#%% md
# #### 2.1. Scaling/ Standardizing Data
#%%
from sklearn.preprocessing import StandardScaler

std = StandardScaler()
X_tr = std.fit_transform(X.values)
X_te = std.transform(X_test.values)
#%% md
# #### 2.2 Ridge Modelling
#%%
from sklearn.linear_model import RidgeCV

n_alphas = 100
alphas = 10 ** np.linspace(-3, 3, n_alphas)

ridge_model = RidgeCV(alphas=alphas, cv=5)

ridge_model.fit(X_tr, y)
print(f"Cross Val R2: {ridge_model.score(X_tr, y)}")
print(f"Cross Val MAE: {mae(np.exp(y), np.exp(ridge_model.predict(X_tr)))}")
print(f"Alpha: {ridge_model.alpha_}")
#%%
pd.DataFrame({"features":X.columns,"coef_ridge":ridge_model.coef_,})
#%% md
# #### 2.3 Lasso Modelling
#%%
from sklearn.linear_model import LassoCV

n_alphas = 200
alphas = 10 ** np.linspace(-2, 3, n_alphas)

lasso_model = LassoCV(alphas=alphas, cv=5)

lasso_model.fit(X_tr, y)

print(f"Cross Val R2: {lasso_model.score(X_tr, y)}")
print(f"Cross Val MAE: {mae(np.exp(y), np.exp(lasso_model.predict(X_tr)))}")
print(f"Alpha: {lasso_model.alpha_}")
#%%
pd.DataFrame({"features":X.columns,"coef_ridge":ridge_model.coef_,
             "coef_lasso":lasso_model.coef_})
#%%
print(mae(np.exp(y_test), np.exp(lasso_model.predict(X_te))))
print(f"Test R2: {r2(y_test, lasso_model.predict(X_te))}")
#%% md
# #### 2.4 Elastic Net Modelling
#%%
from sklearn.linear_model import ElasticNetCV

alphas = 10 ** np.linspace(-2, 3, 200)
l1_ratios = np.linspace(.01, 1, 100)

enet_model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=5)

enet_model.fit(X_tr, y)

print(f"Cross Val R2: {enet_model.score(X_tr, y)}")
print(f"Cross Val MAE: {mae(np.exp(y), np.exp(enet_model.predict(X_tr)))}")
print(f"Alpha: {enet_model.alpha_}")
print(f"L1_Ratio: {enet_model.l1_ratio_}")
#%%
pd.DataFrame({"features":X.columns,"coef_ridge":ridge_model.coef_,
             "coef_lasso":lasso_model.coef_,
             "elastic_net_coef":enet_model.coef_})
#%% md
# #### 2.5 Final Test Modelling
#%%
print(f"Test MAE: {mae(np.exp(y_test), np.exp(ridge_model.predict(X_te)))}")
print(f"Test R2: {r2(y_test, ridge_model.predict(X_te))}")
#%%
