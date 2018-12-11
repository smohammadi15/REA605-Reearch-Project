#!/usr/bin/python3
import os
import glob
import pandas as pd
import numpy as np
#import ipaddress
#from datetime import datetime
#from sklearn.model_selection import train_test_split

#csvlist = glob.glob("/home/user/Downloads/Flowmeter_CSVs/*")
#csvnames = []
#for x in csvlist:
#	y = x.split("/")
#	csvname = y[5]
#	csvnames.append(csvname)
#outof=len(csvnames)
#print(outof)

classifier='lock'
cleancsv = "./Flowmeter_CSVs/clean_lock.csv"
csvlist = "/home/user/Downloads/Flowmeter_CSVs/Lock/*.csv"
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', csvlist))))




#df["Timestamp"]= pd.to_datetime(df.Timestamp)
#df.rename(columns={'Src IP':'SrcIP','Dst IP':'DstIP'})
#print(df.head(5)
print(df.shape)
#print(df.shape[1])
#print(df.columns.tolist())
#df=pd.read_csv("/home/user/Downloads/Flowmeter_CSVs/" +csvnames)

#null_counts = df.isnull().sum()
#print("Number of null values in each column:\n{}".format(null_counts))

#print value and occurence in a column
#print(df["Flow ID"].value_counts())

drop_cols = ['Timestamp','Src IP','Dst IP','Label','Flow ID','Protocol']
df = df.drop(drop_cols,axis=1)

#original features dropped
#drop_cols = ['Timestamp','Src IP','Dst IP','Label','Flow ID','Protocol','Active Mean','Active Min','Active Max','Idle Mean','Idle Max','Idle Min','PSH Flag Cnt','ACK Flag Cnt','Bwd PSH Flags']
#drop_cols = ['Flow ID','Src IP','Dst IP','Timestamp','Protocol','Src Port','Dst Port','Pkt Len Min','Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','Fwd Pkt Len Max','Fwd Pkt Len Min','Fwd Pkt Len Std','Bwd Pkt Len Max','Bwd Pkt Len Min','Flow Pkts/s','PSH Flag Cnt','ACK Flag Cnt','Active Mean','Active Max','Active Min','Idle Mean','Idle Max','Idle Min','Fwd IAT Tot','Fwd IAT Mean','Fwd IAT Std','Fwd IAT Max','Fwd IAT Min','Bwd IAT Tot','Bwd IAT Mean','Bwd IAT Std','Bwd IAT Max','Bwd IAT Min','Fwd Act Data Pkts','Bwd Pkt Len Std','Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min','Bwd PSH Flags','Pkt Len Max','Fwd Header Len','Bwd Header Len','Pkt Len Mean','Pkt Len Std','Pkt Len Var','Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg','Bwd Seg Size Avg','Subflow Fwd Pkts','Subflow Bwd Pkts','Fwd Pkt Len Mean','Bwd Pkt Len Mean']
#df = df.drop(drop_cols,axis=1)
#remove all columns containing missing values
#df = df.dropna()
#print(df.shape)

#get count of column data types
#print("Data types and their frequency\n{}".format(df.dtypes.value_counts()))

#display categorical columns
#object_columns_df = df.select_dtypes(include=['object'])
#print(object_columns_df.iloc[0])

#print(df.head())
#df['SrcIP']= df['SrcIp'].ipaddress.ip_address()
#df["Dst IP"]= pd.to_datetime(df.Dst IP)

#print(df[:10])
#print(df["Idle Max"].value_counts())

#remove columns with no variance; 84->64
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
#print(df.shape)

#print columns with less than 4 unique values
#for col in df.columns:
#    if (len(df[col].unique()) < 4):
#        print(df[col].value_counts())
#        print()

print(df.shape)
#print(df.shape[1])
#print(df.head())

#print()
#print(df.info())

#format column names to lower_case
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

df['classifier']=classifier

df.to_csv(cleancsv,index=False)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)
