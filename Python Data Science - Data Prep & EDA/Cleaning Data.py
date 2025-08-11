import pandas as pd

df = pd.read_excel('/Users/AnirbanGuha/Library/CloudStorage/OneDrive-Personal/Udemy Courses'
                   '/Data Science - data Prep and EDA with Python'
                   '/Data+Science+in+Python+-+Data+Prep+%26+EDA/Data/Alarm Survey Data.xlsx')

df = pd.to_numeric(df.alarm_rating.str.replace(' stars', ''))

print(df.info())
