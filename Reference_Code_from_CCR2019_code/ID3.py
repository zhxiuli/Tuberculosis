
from sklearn import tree
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.externals import joblib
import graphviz

df = pd.read_csv("id3.txt")
x = df.ix[:,:4]
y = df.ix[:,-1]

dtc = tree.DecisionTreeClassifier(criterion="entropy")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = dtc.fit(x_train, y_train)

​dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("clf_1", view=True)

y_test_preds = clf.predict(x_test)

print(clf.feature_importances_)

joblib.dump(clf, "train_model.m")

clf = joblib.load("train_model.m")

clf.predit(test_X)

