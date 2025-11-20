#%% md
# ### Sort and Rank Functions
#%%
import pandas as pd
#%%
Cardata = { "Mercedes": [2, 4, 0, 4, 0, 3], "Ford": [3, 0, 0, 1, 6, 12], "Tata":[9, 3, 4, 1, 0, 0], "Renault":[12, 1, 0, 0, 3, 1]}
Carsales = pd.DataFrame(Cardata)
Carsales.index.rename("Sales place", inplace=True)
Carsales.rename(index={0: "One", 1: "Two", 2: "Three", 3: "Four", 4: "Five", 5: "Six"}, inplace=True)
Carsales
#%%
#Sort the DataFrame using the sort_vales() function
print(Carsales.sort_values(by="Ford", ascending=True))
#%%
#Sort the DataFrame using the sort_vales() function descending order
print(Carsales.sort_values(by="Ford", ascending=False))
#%% md
# #### Sort by multiple columns
#%%
print(Carsales.sort_values(by=["Ford", "Tata"],ascending=False))
#%%
print(Carsales.sort_values(by=["Ford", "Tata"],ascending=[False,True]))
#%%
print(Carsales.sort_index())
#%% md
# ### Find max value for every column using idxmax() function
#%%
print(Carsales.idxmax())
#%%
print(Carsales.idxmax(axis=1))
#%% md
# ### rows value or sales places having minimum value
#%%
print(Carsales.idxmin())
#%%
print(Carsales.idxmin(axis="columns"))
#%%
Carsales.iloc[::2,:]
#%%
#Find the Top 3 highest values for Mercedes and keep duplicates
print(Carsales.nlargest(3,columns="Mercedes",keep="all"))
#%%
#Find the Top 3 highest values for Mercedes and keep duplicates
print(Carsales.nsmallest(3,columns="Mercedes",keep="all"))
#%%
#Rank the sales places by single column
print(Carsales["Mercedes"].rank(ascending=False))
print(Carsales.rank(ascending=False))

#Rank the columns per Sales Place
print(Carsales.rank(axis=1,ascending=False))

