import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("../../data/Online Retail.xlsx")

df = df.dropna(subset=['CustomerID'])

df  = df[
    (df['Quantity'] > 0) &
    (df['UnitPrice'] > 0)
]

df["TotalSpend"] = df['Quantity'] * df['UnitPrice']

#group by customer id

customer_df = df.groupby("CustomerID").agg({
    'Quantity' : 'sum',
    "UnitPrice" : 'mean',
    "TotalSpend" : 'sum',
    'Country' : 'first'
    }).reset_index()

threshold = customer_df['TotalSpend'].median()

customer_df['high_value'] = (customer_df['TotalSpend'] > threshold).astype(int)

#Convert the county in to numbers so we can use it.
label = LabelEncoder()
customer_df['Country'] = label.fit_transform(customer_df["Country"])

#Now we need to choose the x any y

x = customer_df[["Quantity","UnitPrice","Country"]]
y = customer_df['high_value']

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=50,
    random_state=42
)

model.fit(x,y)

plt.figure(figsize=(18,8))

plot_tree(
    model,
    feature_names=x.columns,
    class_names=['low','high'],
    filled=True,
    rounded=True
)

plt.title("Customer segmentation using Decision tree")
plt.show()

importance = pd.Series(
    model.feature_importances_,
    index=x.columns
)

print("Feature importance")
print(importance.sort_values(ascending=False))