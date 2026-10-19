import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# LOAD DATASET
retail = pd.read_csv(
    r"C:\Users\c3698852\Downloads\Online Retail.csv",
    encoding="latin1"
)

# FIX COLUMN NAMES
retail.columns = retail.columns.str.replace('ï»¿', '', regex=False)
retail.columns = retail.columns.str.strip()

# REMOVE CANCELLED ORDERS
retail = retail[
    ~retail['InvoiceNo'].astype(str).str.contains('C', na=False)
]

# REMOVE MISSING DESCRIPTIONS
retail.dropna(subset=['Description'], inplace=True)

# CREATE BASKET FOR FRANCE
basket_France = (
    retail[retail['Country'] == 'France']
    .groupby(['InvoiceNo', 'Description'])['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)

# HOT ENCODING
basket_France = (basket_France > 0).astype(int)

# APRIORI ALGORITHM
frequent_itemsets = apriori(
    basket_France,
    min_support=0.07,
    use_colnames=True
)

print("\n========== FREQUENT ITEMSETS ==========")
print(frequent_itemsets)

# ASSOCIATION RULES
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

print("\n========== ASSOCIATION RULES ==========")
print(
    rules[
        [
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]
    ]
)

# STRONG RULES
strong_rules = rules[
    (rules['lift'] >= 3) &
    (rules['confidence'] >= 0.3)
]

print("\n========== STRONG RULES ==========")
print(
    strong_rules[
        [
            'antecedents',
            'consequents',
            'support',
            'confidence',
            'lift'
        ]
    ]
)