# -*- coding: utf-8 -*-
#from __future__ import unicode_literals


from sklearn import datasets
scom = datasets.load_iris()

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

from flask import Flask, request
from flask_restful import Resource, Api



app = Flask(__name__)
api = Api(app)

class previsione8(Resource):
    def get(self,progetto,v1,p1,s1,r1,v2,p2,s2,r2):

        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        #print("------------------------------------------")
        predizione2 = clf.predict_proba([[v1, p1, s1, r1, v2, p2, s2, r2]])
        #print(str(squadra1)+" vs "+str(squadra2)+"    1: "+str(predizione2[0,0]*100)+"% --- X: "+str(predizione2[0,2]*100)+"% --- 2: "+str(predizione2[0,1]*100)+"%")
        return str(predizione2[0,])


class previsione1(Resource):
    def get(self,progetto,v1):

        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        #y_pred = clf.predict(X_test)
        #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        #print("------------------------------------------")
        predizione2 = clf.predict_proba([[v1, 0, 0, 0, 0, 0, 0, 0]])
        #print(str(squadra1)+" vs "+str(squadra2)+"    1: "+str(predizione2[0,0]*100)+"% --- X: "+str(predizione2[0,2]*100)+"% --- 2: "+str(predizione2[0,1]*100)+"%")
        return str(predizione2[0,])



class previsione4(Resource):
    def get(self,progetto, v1, p1, s1, r1):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, s1, r1, 0, 0, 0, 0]])

        return str(predizione2[0,])


class previsione2(Resource):
    def get(self,progetto, v1, p1):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, 0, 0, 0, 0, 0, 0]])

        return str(predizione2[0,])


class previsione3(Resource):
    def get(self,progetto, v1, p1, s1):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, s1, 0, 0, 0, 0, 0]])

        return str(predizione2[0,])


class previsione5(Resource):
    def get(self,progetto, v1, p1, s1, r1, v2):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, s1, r1, v2, 0, 0, 0]])

        return str(predizione2[0,])


class previsione6(Resource):
    def get(self,progetto, v1, p1, s1, r1, v2, p2):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, s1, r1, v2, p2, 0, 0]])

        return str(predizione2[0,])


class previsione7(Resource):
    def get(self,progetto, v1, p1, s1, r1, v2, p2, s2):
        data = pd.DataFrame({
            'v1': scom.data[:, 0],
            'p1': scom.data[:, 1],
            's1': scom.data[:, 2],
            'r1': scom.data[:, 3],
            'v2': scom.data[:, 4],
            'p2': scom.data[:, 5],
            's2': scom.data[:, 6],
            'r2': scom.data[:, 7],
            'ris': scom.target
        })
        data.head()

        X = data[['v1', 'p1', 's1', 'r1', 'v2', 'p2', 's2', 'r2']]  # Features
        y = data['ris']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        # y_pred = clf.predict(X_test)
        # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        predizione2 = clf.predict_proba([[v1, p1, s1, r1, v2, p2, s2, 0]])

        return str(predizione2[0,])


api.add_resource(previsione1, '/previsione/<progetto>&<v1>') # Route_1
api.add_resource(previsione2, '/previsione/<progetto>&<v1>&<p1>') # Route_2
api.add_resource(previsione3, '/previsione/<progetto>&<v1>&<p1>&<s1>') # Route_3
api.add_resource(previsione4, '/previsione/<progetto>&<v1>&<p1>&<s1>&<r1>') # Route_4
api.add_resource(previsione5, '/previsione/<progetto>&<v1>&<p1>&<s1>&<r1>&<v2>') # Route_5
api.add_resource(previsione6, '/previsione/<progetto>&<v1>&<p1>&<s1>&<r1>&<v2>&<p2>') # Route_6
api.add_resource(previsione7, '/previsione/<progetto>&<v1>&<p1>&<s1>&<r1>&<v2>&<p2>&<s2>') # Route_7
api.add_resource(previsione8, '/previsione/<progetto>&<v1>&<p1>&<s1>&<r1>&<v2>&<p2>&<s2>&<r2>') # Route_8

if __name__ == '__main__':
     app.run(port='5002')
