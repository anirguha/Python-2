import pandas as pd
#%%
Cardata = { "Mercedes": [2, 4, 0, 3, 0, 3], "Ford": [3, 0, 0, 1, 6, 12], "Tata":[9, 3, 4, 1, 0, 0], "Renault":[12, 1, 0, 0, 3, 1]}
Carsales = pd.DataFrame(Cardata)
Carsales.index.rename("Sales place", inplace=True)
Carsales.rename(index={0: "One", 1: "Two", 2: "Three", 3: "Four", 4: "Five", 5: "Six"}, inplace=True)
print(Carsales)

print(Carsales.agg(["sum","min","max","mean","median","std"]))

#Create summary description for selected columns
print(Carsales.agg(Sum=("Ford","sum"),Maximum=("Ford","max"),Minimum=("Ford","min")))

#Create different aggregates for selected column
print(Carsales.agg({"Ford":["sum","min","max","mean","median","std"],"Tata":["sum","min","max","mean","median","std","count"]}))

#Change types of values
Carsales = Carsales.astype(int)
print(Carsales)

#Create same aggregate for multiple columns
print(Carsales[["Ford","Tata","Mercedes"]].agg("sum"))

#Create aggregate for rows
print(Carsales.agg("mean",axis="columns"))

#Use Descriptive statistics
print(Carsales["Ford"].describe())

#Use Value Count Function
print(Carsales.value_counts())

#Normalize the values to show distribution
print(Carsales["Ford"].value_counts(normalize=True))
print(Carsales.value_counts(normalize=True))

#Add a column to sum for each row
Carsales["Total Sales"] = Carsales[Carsales.columns].sum(axis=1)
print(Carsales)

#Use pandas apply function to create a new column based on values of one/other columns
def sales_perf(num_sold):
    if num_sold >= 15:
        return "Excellent"
    elif num_sold >= 5:
        return "Acceptable"
    else:
        return "Poor"

Carsales["Sales Perf"] = Carsales["Total Sales"].apply(sales_perf)
print(Carsales)
