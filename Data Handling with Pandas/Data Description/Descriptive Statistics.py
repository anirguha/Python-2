import pandas as pd
#%%
Cardata = { "Mercedes": [2, 4, 0, 3, 0, 3], "Ford": [3, 0, 0, 1, 6, 12], "Tata":[9, 3, 4, 1, 0, 0], "Renault":[12, 1, 0, 0, 3, 1]}
Carsales = pd.DataFrame(Cardata)
Carsales.index.rename("Sales place", inplace=True)
Carsales.rename(index={0: "One", 1: "Two", 2: "Three", 3: "Four", 4: "Five", 5: "Six"}, inplace=True)
print(Carsales)

print(Carsales.agg(["sum","min","max","mean","median","std"]))
