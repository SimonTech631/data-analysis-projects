import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load/Read the data
titan =pd.read_csv('titan_train.csv')
titan.head(20)
titan.tail(30)

#Find the missing values from the dataset
titan.isnull()

#Find the total for all the missing values in the dataset
titan.isnull().sum()

#Plot a graph from the total of all the missing values in the dataset
titan.isnull().sum().plot()

#2. Set style
sns.set_theme(style="whitegrid")

#3. Plot 1: Age by Class
plt.title('Age Distribution by Passenger Class')
plt.figure(figsize=(10,6))
sns.boxplot(data=titan, x='Pclass', y='Age')

#3. Plot 2: Survival by Sex
plt.title('Survival Count by Sex')
plt.figure(figsize=(8,5))
sns.countplot(data=titan, x='Sex', hue='Survived')

#4. plot 3: Correlation Heatmap
plt.title('Correlation Heatmap')
plt.figure(figsize=(10,8))
sns.heatmap(titan.corr(numeric_only=True), annot=False, camp='coolwarm')
