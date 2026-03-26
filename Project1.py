import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv(r'C:\Users\HP\OneDrive\Attachments\Desktop\Machine learning Notes\insurance.csv')
# print(data)
#EDA
# print(data.shape)
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())
# print(data.columns)
# numeric_columns = ['age','bmi','children','charges']
# for col in numeric_columns:
    # plt.figure(figsize=(6,4))
    # sns.histplot(data[col], kde = True, bins = 20)
    # # sns.countplot(x = data['children'])
    # # sns.countplot(x = data['sex'])
    # sns.countplot(x = data['smoker'])
    # plt.show()
    # sns.boxplot(x = data[col])
    # plt.show()
    # plt.figure( figsize= (8,6))
    # sns.heatmap(data.corr(numeric_only= True), annot= True)
    # plt.show()


"Data Cleaning and Preprocessing"

df_cleaned = data.copy()
# df_cleaned.head()
# df_cleaned.shape
# df_cleaned.drop_duplicates(inplace= True)
# df_cleaned.shape
# df_cleaned.isnull().sum()
# print(df_cleaned)
print(df_cleaned['sex'].value_counts())
df_cleaned['sex'] = df_cleaned['sex'].map({"male":0, "female":1})
print(df_cleaned['smoker'].value_counts())
df_cleaned['smoker'] = df_cleaned['smoker'].map({"no":0, "yes":1})
df_cleaned.rename(columns={'sex':'is_female','smoker':'is_smoker'},inplace=True)
print(df_cleaned['region'].value_counts())
df_cleaned = pd.get_dummies(df_cleaned, columns=['region'], drop_first=True)
df_cleaned = df_cleaned.astype(int)
print(df_cleaned)

