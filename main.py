#START OF FIT3164 CODE
#Prepared by Prashant Murali, Jess Nalder, Kevin Khuu

#Load dataset
import pandas as pd
dataset = pd.read_csv("CAD4.csv")

#Convert dataset to dataframe
df=pd.DataFrame(dataset)
print(df)
#View the datatypes and summary
print(df.dtypes)
print(df.describe())

print("\n")

#Check if there is any null values in the dataframe
total_empty_values=(df.isnull().sum().sum())
print("Number of missing or null values in the dataframe:"+str(total_empty_values))
if total_empty_values==0:
    print("Data imputation is not needed.")
else:
    print("Data imputation is required to resolve null values in the dataframe.")
    #Note that there are many packages that offer data imputation methods in the event that the
    # dataset is not of a satisfactory level in terms of data quality/cleanliness.