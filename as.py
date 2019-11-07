import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import json


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))

values_dict = dict({
   "cap-shape":{
      "b":"bell",
      "c":"conical",
      "x":"convex",
      "f":"flat",
      "k":"knobbed",
      "s":"sunken"
   },
   "cap-surface":{
      "f":"fibrous",
      "g":"grooves",
      "y":"scaly",
      "s":"smooth"
   },
   "cap-color":{
      "n":"brown",
      "b":"buff",
      "c":"cinnamon",
      "g":"gray",
      "r":"green",
      "p":"pink",
      "u":"purple",
      "e":"red",
      "w":"white",
      "y":"yellow"
   },
   "bruises?":{
      "t":"bruises",
      "f":"no"
   },
   "odor":{
      "a":"almond",
      "l":"anise",
      "c":"creosote",
      "y":"fishy",
      "f":"foul",
      "m":"musty",
      "n":"none",
      "p":"pungent",
      "s":"spicy"
   },
   "gill-attachment":{
      "a":"attached",
      "d":"descending",
      "f":"free",
      "n":"notched"
   },
   "gill-spacing":{
      "c":"close",
      "w":"crowded",
      "d":"distant"
   },
   "gill-size":{
      "b":"broad",
      "n":"narrow"
   },
   "gill-color":{
      "k":"black",
      "n":"brown",
      "b":"buff",
      "h":"chocolate",
      "g":"gray",
      "r":"green",
      "o":"orange",
      "p":"pink",
      "u":"purple",
      "e":"red",
      "w":"white",
      "y":"yellow"
   },
   "stalk-shape":{
      "e":"enlarging",
      "t":"tapering"
   },
   "stalk-root":{
      "b":"bulbous",
      "c":"club",
      "u":"cup",
      "e":"equal",
      "z":"rhizomorphs",
      "r":"rooted",
      "?":"missing"
   },
   "stalk-surface-above-ring":{
      "f":"fibrous",
      "y":"scaly",
      "k":"silky",
      "s":"smooth"
   },
   "stalk-surface-below-ring":{
      "f":"fibrous",
      "y":"scaly",
      "k":"silky",
      "s":"smooth"
   },
   "stalk-color-above-ring":{
      "n":"brown",
      "b":"buff",
      "c":"cinnamon",
      "g":"gray",
      "o":"orange",
      "p":"pink",
      "e":"red",
      "w":"white",
      "y":"yellow"
   },
   "stalk-color-below-ring":{
      "n":"brown",
      "b":"buff",
      "c":"cinnamon",
      "g":"gray",
      "o":"orange",
      "p":"pink",
      "e":"red",
      "w":"white",
      "y":"yellow"
   },
   "veil-type":{
      "p":"partial",
      "u":"universal"
   },
   "veil-color":{
      "n":"brown",
      "o":"orange",
      "w":"white",
      "y":"yellow"
   },
   "ring-number":{
      "n":"none",
      "o":"one",
      "t":"two"
   },
   "ring-type":{
      "c":"cobwebby",
      "e":"evanescent",
      "f":"flaring",
      "l":"large",
      "n":"none",
      "p":"pendant",
      "s":"sheathing",
      "z":"zone"
   },
   "spore-print-color":{
      "k":"black",
      "n":"brown",
      "b":"buff",
      "h":"chocolate",
      "r":"green",
      "o":"orange",
      "u":"purple",
      "w":"white",
      "y":"yellow"
   },
   "population":{
      "a":"abundant",
      "c":"clustered",
      "n":"numerous",
      "s":"scattered",
      "v":"several",
      "y":"solitary"
   },
   "habitat":{
      "g":"grasses",
      "l":"leaves",
      "m":"meadows",
      "p":"paths",
      "u":"urban",
      "w":"waste",
      "d":"woods"
   }
})
# print(json.dumps(values_dict, indent=2))

column_names = ['class',
                'cap-shape',
                'cap-surface',
                'cap-color',
                'bruises?',
                'odor',
                'gill-attachment',
                'gill-spacing',
                'gill-size',
                'gill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']
# data = pd.read_csv('data-translated.csv', delimiter=";")

data = pd.read_csv('agaricus-lepiota.data', header=None, names=column_names)

print(data.head(5))
data_without_missing_column = data.drop(columns=['stalk-root'])
del values_dict['stalk-root']
print(data_without_missing_column.head(5))

# can be changed between original/without_missing
df = data_without_missing_column
# df = data

# --------------------------feature selection-----------------------------
# encode categorical variable
labelencoder = LabelEncoder()
mapping_dict = {}
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
    mapping_dict[column] = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))

print(df.describe())
print("---------mapping-------------")
# print(json.dumps(mapping_dict, indent=2))


final_dict = {}
for col_name in values_dict:
    final_dict[col_name] ={}
    for key in mapping_dict[col_name]:
        final_dict[col_name][mapping_dict[col_name][key]] = values_dict[col_name][key]

print("--------- final mapping-------------")
print(json.dumps(final_dict, indent=2))

    # column has the same value in all records
df = df.drop(["veil-type"], axis=1)

# characteristics
df_div = pd.melt(df, "class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(10, 5))
p = sns.violinplot(ax=ax, x="Characteristics", y="value", hue="class", split=True, data=df_div, inner='quartile',
                   palette='Set1')
df_no_class = df.drop(["class"], axis=1)
p.set_xticklabels(rotation=90, labels=list(df_no_class.columns))

# edible/poisonous plot
pd.Series(df['class']).value_counts().sort_index().plot(kind='bar')
plt.ylabel("Count")
plt.xlabel("class")
plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)');

# correlation
plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), linewidths=.1, cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);

# least correlated is gill-color
print(
    df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False))

# analysis on gill-color
new_var = df[['class', 'gill-color']]
new_var = new_var[new_var['gill-color'] <= 3.5]
sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);

new_var = df[['class', 'gill-color']]
new_var = new_var[new_var['gill-color'] > 3.5]
sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);

# building model
X = df.drop(['class'], axis=1)
Y = df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Decision tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

dot_data = export_graphviz(clf, out_file=None,
                           feature_names=X.columns,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render()

# feature importance
features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(5, 7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
plt.show()
