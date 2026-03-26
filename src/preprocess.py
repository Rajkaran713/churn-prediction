import pandas as pd
from sklearn.model_selection import train_test_split

def data_loading_1(filepath_1):
    df1=pd.read_csv(filepath_1)
    return df1

def data_loading_2(filepath_2):
    df2=pd.read_csv(filepath_2)
    return df2

def data_concatenation(df1,df2):
    df=pd.concat([df1,df2],axis=0)
    df = df.reset_index(drop=True)
    return df

def preprocess(df):

    #dropping high cardinality and multicolinearity features
    drop_columns=['State','Total eve charge','Total day charge','Total night charge','Total intl charge']
    df=df.drop(columns=drop_columns)

    #encoding categorical features
    df['International plan']=df['International plan'].map({'Yes':1,'No':0})
    df['Voice mail plan']=df['Voice mail plan'].map({'Yes':1,'No':0})
    df['Churn']=df['Churn'].map({True:1,False:0})

    return df

def data_splitting(df):
    X=df.drop(columns="Churn")
    Y=df['Churn']

    X_Train, X_Test, Y_Train, Y_Test= train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=42)

    return X_Train, X_Test, Y_Train, Y_Test
