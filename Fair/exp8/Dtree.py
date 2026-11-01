
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# 1. Load dataset
df = pd.read_excel("../../data/Online Retail.xlsx")

# 2. Preprocessing
df = df.dropna(subset=['CustomerID'])

df = df[
    (df['Quantity'] > 0) &
    (df['UnitPrice'] > 0)
]

# 3. Calculate total spend
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# 4. Group data by customer
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'UnitPrice': 'mean',
    'TotalSpend': 'sum',
    'Country': 'first'
}).reset_index()

# 5. Create target variable
threshold = customer_df['TotalSpend'].median()

customer_df['HighValue'] = (
    customer_df['TotalSpend'] > threshold
).astype(int)

# 6. Encode country
label = LabelEncoder()

customer_df['Country'] = label.fit_transform(
    customer_df['Country']
)

# Select features and target
X = customer_df[['Quantity', 'UnitPrice', 'Country']]
y = customer_df['HighValue']

# 7. Train entropy-based decision tree
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

model.fit(X, y)

# 8. Visualize decision tree
plt.figure(figsize=(18, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Low', 'High'],
    filled=True,
    rounded=True
)

plt.title("Customer Segmentation using Decision Tree")
plt.show()

# 9. Display feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

print("Feature Importance:")
print(importance.sort_values(ascending=False))
