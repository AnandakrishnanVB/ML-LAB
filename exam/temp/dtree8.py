import pandas as pd
import matplotlib.pyploy as plt

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("../../data/Online Retail.xlsx")

df = df.dropna(subset=['CustomerID'])

df  = df[
    (df['Quantity'] > 0) &
    (df['Unitprice'])
]

df["TotalSpend"] = df['Quantity'] * df['Unitprice']

#group by customer id

customer_df = df.groupby("CustomerID").agg({
    'Quantity' : 'sum',
    "UnitPrice" : 'mean',
    "TotalSpend" : 'sum'
}).reset_index()

print(customer_df)