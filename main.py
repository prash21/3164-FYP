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


#convert "sex" column into 1,0
count = 0
#0 for male, 1 for female
for row in df['Sex']:
    if row == "Male":
        df.at[count,'Sex']= 0
    if row == "Fmale":
        df.at[count, 'Sex'] = 1
    count += 1

df['Sex'] = pd.to_numeric(df['Sex'])



#convert "Cath" column into 1,0
count = 0
#0 for Normal, 1 for Cad
for row in df['Cath']:
    if row == "Normal":
        df.at[count,'Cath']= 0
    if row == "Cad":
        df.at[count, 'Cath'] = 1
    count += 1

df['Cath'] = pd.to_numeric(df['Cath'])




#convert other columns to 1,0
def convertCategorical(df, column):
    count = 0
    # 0 for NO, 1 for YES
    for row in df[column]:
        if row == "N":
            df.at[count, column] = 0
        if row == "Y":
            df.at[count, column] = 1
        count += 1

    df[column] = pd.to_numeric(df[column])

#Call function on all other columns that have Y/N values to convert them into
# 0/1

#Convert all Y/N columns to 1/0
convertCategorical(df, "Obesity")
convertCategorical(df, "CRF")
convertCategorical(df, "CVA")
convertCategorical(df, "Airway disease")
convertCategorical(df, "Thyroid Disease")
convertCategorical(df, "CHF")
convertCategorical(df, "DLP")
convertCategorical(df, "Weak Peripheral Pulse")
convertCategorical(df, "Lung rales")
convertCategorical(df, "Systolic Murmur")
convertCategorical(df, "Diastolic Murmur")
convertCategorical(df, "Dyspnea")
convertCategorical(df, "Atypical")
convertCategorical(df, "Nonanginal")
convertCategorical(df, "Exertional CP")
convertCategorical(df, "LowTH Ang")
convertCategorical(df, "LVH")
convertCategorical(df, "Poor R Progression")



#Convert MALE/FEMALE into separate columns
df["Male"]=""
df["Female"]=""

count = 0
# 0 for NO, 1 for YES
for row in df["Sex"]:
    if row == 0:
        df.at[count, "Male"] = 1
        df.at[count, "Female"] = 0
    if row == 1:
        df.at[count, "Male"] = 0
        df.at[count, "Female"] = 1
    count += 1

df["Male"] = pd.to_numeric(df["Male"])
df["Female"] = pd.to_numeric(df["Female"])

df=df.drop(['Sex'], axis=1)



#Convert function class into separate columns
df["Function Class 0"]=""
df["Function Class 1"]=""
df["Function Class 2"]=""
df["Function Class 3"]=""
df["Function Class 4"]=""

count = 0
# 0 for NO, 1 for YES
for row in df["Function Class"]:
    if row == 0:
        df.at[count, "Function Class 0"] = 1
        df.at[count, "Function Class 1"] = 0
        df.at[count, "Function Class 2"] = 0
        df.at[count, "Function Class 3"] = 0
        df.at[count, "Function Class 4"] = 0
    if row == 1:
        df.at[count, "Function Class 0"] = 0
        df.at[count, "Function Class 1"] = 1
        df.at[count, "Function Class 2"] = 0
        df.at[count, "Function Class 3"] = 0
        df.at[count, "Function Class 4"] = 0
    if row == 2:
        df.at[count, "Function Class 0"] = 0
        df.at[count, "Function Class 1"] = 0
        df.at[count, "Function Class 2"] = 1
        df.at[count, "Function Class 3"] = 0
        df.at[count, "Function Class 4"] = 0
    if row == 3:
        df.at[count, "Function Class 0"] = 0
        df.at[count, "Function Class 1"] = 0
        df.at[count, "Function Class 2"] = 0
        df.at[count, "Function Class 3"] = 1
        df.at[count, "Function Class 4"] = 0
    if row == 4:
        df.at[count, "Function Class 0"] = 0
        df.at[count, "Function Class 1"] = 0
        df.at[count, "Function Class 2"] = 0
        df.at[count, "Function Class 3"] = 0
        df.at[count, "Function Class 4"] = 1
    count += 1

df["Function Class 0"] = pd.to_numeric(df["Function Class 0"])
df["Function Class 1"] = pd.to_numeric(df["Function Class 1"])
df["Function Class 2"] = pd.to_numeric(df["Function Class 2"])
df["Function Class 3"] = pd.to_numeric(df["Function Class 3"])
df["Function Class 4"] = pd.to_numeric(df["Function Class 4"])

df=df.drop(['Function Class'], axis=1)



#Convert BBB into separate columns
df["BBB_LBBB"]=""
df["BBB_N"]=""
df["BBB_RBBB"]=""

count = 0
# 0 for NO, 1 for YES
for row in df["BBB"]:
    if row == "LBBB":
        df.at[count, "BBB_LBBB"] = 1
        df.at[count, "BBB_N"] = 0
        df.at[count, "BBB_RBBB"] = 0
    if row == "N":
        df.at[count, "BBB_LBBB"] = 0
        df.at[count, "BBB_N"] = 1
        df.at[count, "BBB_RBBB"] = 0
    if row == "RBBB":
        df.at[count, "BBB_LBBB"] = 0
        df.at[count, "BBB_N"] = 0
        df.at[count, "BBB_RBBB"] = 1

    count += 1

df["BBB_LBBB"]=pd.to_numeric(df["BBB_LBBB"])
df["BBB_N"]=pd.to_numeric(df["BBB_N"])
df["BBB_RBBB"]=pd.to_numeric(df["BBB_RBBB"])

df=df.drop(['BBB'], axis=1)



#Convert function class into separate columns
df["Region RWMA 0"]=""
df["Region RWMA 1"]=""
df["Region RWMA 2"]=""
df["Region RWMA 3"]=""
df["Region RWMA 4"]=""

count = 0
# 0 for NO, 1 for YES
for row in df["Region RWMA"]:
    if row == 0:
        df.at[count, "Region RWMA 0"] = 1
        df.at[count, "Region RWMA 1"] = 0
        df.at[count, "Region RWMA 2"] = 0
        df.at[count, "Region RWMA 3"] = 0
        df.at[count, "Region RWMA 4"] = 0
    if row == 1:
        df.at[count, "Region RWMA 0"] = 0
        df.at[count, "Region RWMA 1"] = 1
        df.at[count, "Region RWMA 2"] = 0
        df.at[count, "Region RWMA 3"] = 0
        df.at[count, "Region RWMA 4"] = 0
    if row == 2:
        df.at[count, "Region RWMA 0"] = 0
        df.at[count, "Region RWMA 1"] = 0
        df.at[count, "Region RWMA 2"] = 1
        df.at[count, "Region RWMA 3"] = 0
        df.at[count, "Region RWMA 4"] = 0
    if row == 3:
        df.at[count, "Region RWMA 0"] = 0
        df.at[count, "Region RWMA 1"] = 0
        df.at[count, "Region RWMA 2"] = 0
        df.at[count, "Region RWMA 3"] = 1
        df.at[count, "Region RWMA 4"] = 0
    if row == 4:
        df.at[count, "Region RWMA 0"] = 0
        df.at[count, "Region RWMA 1"] = 0
        df.at[count, "Region RWMA 2"] = 0
        df.at[count, "Region RWMA 3"] = 0
        df.at[count, "Region RWMA 4"] = 1
    count += 1

df["Region RWMA 0"]=pd.to_numeric(df["Region RWMA 0"])
df["Region RWMA 1"]=pd.to_numeric(df["Region RWMA 1"])
df["Region RWMA 2"]=pd.to_numeric(df["Region RWMA 2"])
df["Region RWMA 3"]=pd.to_numeric(df["Region RWMA 3"])
df["Region RWMA 4"]=pd.to_numeric(df["Region RWMA 4"])

df=df.drop(['Region RWMA'], axis=1)



#Convert function class into separate columns
df["VHD_N"]=""
df["VHD_Mild"]=""
df["VHD_Severe"]=""
df["VHD_Moderate"]=""

count = 0
# 0 for NO, 1 for YES
for row in df["VHD"]:
    if row == "N":
        df.at[count, "VHD_N"] = 1
        df.at[count, "VHD_Mild"] = 0
        df.at[count, "VHD_Severe"] = 0
        df.at[count, "VHD_Moderate"] = 0
    if row == "mild":
        df.at[count, "VHD_N"] = 0
        df.at[count, "VHD_Mild"] = 1
        df.at[count, "VHD_Severe"] = 0
        df.at[count, "VHD_Moderate"] = 0
    if row == "Severe":
        df.at[count, "VHD_N"] = 0
        df.at[count, "VHD_Mild"] = 0
        df.at[count, "VHD_Severe"] = 1
        df.at[count, "VHD_Moderate"] = 0
    if row == "Moderate":
        df.at[count, "VHD_N"] = 0
        df.at[count, "VHD_Mild"] = 0
        df.at[count, "VHD_Severe"] = 0
        df.at[count, "VHD_Moderate"] = 1

    count += 1

df["VHD_N"]=pd.to_numeric(df["VHD_N"])
df["VHD_Mild"]=pd.to_numeric(df["VHD_Mild"])
df["VHD_Severe"]=pd.to_numeric(df["VHD_Severe"])
df["VHD_Moderate"]=pd.to_numeric(df["VHD_Moderate"])

df=df.drop(['VHD'], axis=1)

#Output the modified dataset to a csv to see it clearly.
df.to_csv("CAD4_Updated.csv")


#Feature selection

#Find correlation between all variables.
correlation_df=df.corr()
print(correlation_df)
correlation_df.to_csv("all_correlation.csv")

#Plot the heatmap for all variables
import matplotlib.pyplot as plt
#Edit the parameters of the heatmap style
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
#Plot the heatmap - UNCOMMENT TO SEE HEATMAP
#plt.matshow(correlation_df.corr())
#plt.show()

#Plot is quite big, so look closer between Cath and all variables.
corr_list=[]
#Print and look at all the correlation values between Cath and all other attributes.
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(correlation_df["Cath"])

#All attributes seem to have some/minor correlation with Cath, so they will be kept.

#here we go