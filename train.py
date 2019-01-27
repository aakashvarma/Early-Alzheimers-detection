import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

df = pd.read_csv('mri-and-alzheimers/oasis_longitudinal.csv')

# print df.head()
# print df.shape

df = df.loc[df['Visit']==1] # use first visit data only because of the analysis we're doing
df = df.reset_index(drop=True) # reset index after filtering first visit data
df['M/F'] = df['M/F'].replace(['F','M'], [0,1]) # M/F column
df['Group'] = df['Group'].replace(['Converted'], ['Demented']) # Target variable
df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1,0]) # Target variable
df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1) # Drop unnecessary columns

# print df.head()
# print df.shape

def bar_chart(feature):
    Demented = df[df['Group']==1][feature].value_counts()
    Nondemented = df[df['Group']==0][feature].value_counts()
    df_bar = pd.DataFrame([Demented,Nondemented])
    df_bar.index = ['Demented','Nondemented']
    df_bar.plot(kind='bar',stacked=True, figsize=(8,5))

bar_chart('M/F')
plt.xlabel('Group')
plt.ylabel('Number of patients')
plt.legend()
plt.title('Gender and Demented rate')

# plt.show()


def analysis():
    #MMSE : Mini Mental State Examination
    # Nondemented = 0, Demented =1
    # Nondemented has higher test result ranging from 25 to 30. 
    #Min 17 ,MAX 30
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'MMSE',shade= True)
    facet.set(xlim=(0, df['MMSE'].max()))
    facet.add_legend()
    plt.xlim(15.30)

    #bar_chart('ASF') = Atlas Scaling Factor
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'ASF',shade= True)
    facet.set(xlim=(0, df['ASF'].max()))
    facet.add_legend()
    plt.xlim(0.5, 2)

    #eTIV = Estimated Total Intracranial Volume
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'eTIV',shade= True)
    facet.set(xlim=(0, df['eTIV'].max()))
    facet.add_legend()
    plt.xlim(900, 2100)

    #'nWBV' = Normalized Whole Brain Volume
    # Nondemented = 0, Demented =1
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'nWBV',shade= True)
    facet.set(xlim=(0, df['nWBV'].max()))
    facet.add_legend()
    plt.xlim(0.6,0.9)

    #AGE. Nondemented =0, Demented =0
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, df['Age'].max()))
    facet.add_legend()
    plt.xlim(50,100)

    #'EDUC' = Years of Education
    # Nondemented = 0, Demented =1
    facet= sns.FacetGrid(df,hue="Group", aspect=3)
    facet.map(sns.kdeplot,'EDUC',shade= True)
    facet.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
    facet.add_legend()
    plt.ylim(0, 0.16)

    plt.show()

# analysis()

# Intermediate Result Summary
# Men are more likely with demented, an Alzheimer's Disease, than Women.
# Demented patients were less educated in terms of years of education.
# Nondemented group has higher brain volume than Demented group.
# Higher concentration of 70-80 years old in Demented group than those in the nondemented patients




























