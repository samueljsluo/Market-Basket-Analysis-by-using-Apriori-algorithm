import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as py
from plotly.offline import plot

dataset = pd.read_csv("data/Groceries_dataset.csv")

print(f'Null Values: \n{dataset.isnull().sum()}\n')
print(f'Nan Values: \n{dataset.isna().sum()}\n')

# chart = px.bar(dataset["itemDescription"].value_counts()[:30], orientation='v',
#                color=dataset['itemDescription'].value_counts()[:30], title='Item by count', labels={'value':'Count',
#                                                                                                     'index':'Item'})
# chart.show()
# all_product = dataset['itemDescription'].unique()
# one_hot = pd.get_dummies(dataset['itemDescription'])
# dataset.drop('itemDescription', inplace=True, axis=1)
# dataset = dataset.join(one_hot)
transaction = dataset.groupby(['Member_number','Date'])['itemDescription'].apply(','.join).reset_index()
transaction['itemDescription'] = transaction['itemDescription'].str.split(',')
association_rules = apriori(list(transaction.itemDescription), min_support=0.00030, min_confidance=0.05, min_lift=3,
                min_length=2, target="rules")
for item in association_rules:
    pair = item[0]
    items = [x for x in pair]

    print("Rule : ", items[0], " -> " + items[1])
    print("Support : ", str(item[1]))
    print("Confidence : ", str(item[2][0][2]))
    print("Lift : ", str(item[2][0][3]))

    print("=====================================")