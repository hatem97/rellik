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

class risultatoRandomForest(Resource):
    def get(self,v1,p1,s1,r1,v2,p2,s2,r2, squadra1, squadra2):

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

        print("------------------------------------------")
        predizione2 = clf.predict_proba([[v1, p1, s1, r1, v2, p2, s2, r2]])
        print(str(squadra1)+" vs "+str(squadra2)+"    1: "+str(predizione2[0,0]*100)+"% --- X: "+str(predizione2[0,2]*100)+"% --- 2: "+str(predizione2[0,1]*100)+"%")
        return str(predizione2[0,])



class risultatoReteNeurale(Resource):
    def get(self,v1,p1,s1,r1,v2,p2,s2,r2, squadra1, squadra2):
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        scaler = StandardScaler()
        scaler.fit(X_train)
        StandardScaler(copy=True, with_mean=True, with_std=True)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=200)
        mlp.fit(X_train, y_train)
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                      beta_2=0.999, early_stopping=False, epsilon=1e-08,
                      hidden_layer_sizes=(6, 3), learning_rate='constant',
                      learning_rate_init=0.001, max_iter=200, momentum=0.9,
                      nesterovs_momentum=True, power_t=0.5, random_state=None,
                      shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                      verbose=False, warm_start=False)
        y_pred = mlp.predict(X_test)
        #print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        #print("previsione: " + str(mlp.predict([[v1, p1, s1, r1, v2, p2, s2, r2]])))
        predictions = mlp.predict_proba([[2, 2, 1, 1, 4, 1, 0, 2]])
        print("previsione "+str(squadra1)+" vs "+str(squadra2)+"   1: " + str(predictions[0, 0] * 100) + "% --- X: " + str(
            predictions[0, 2] * 100) + "% --- 2: " + str(predictions[0, 1] * 100) + "%")
        print(confusion_matrix(y_test, y_pred))

        return str(predictions[0,])


api.add_resource(risultatoRandomForest, '/risultatoRandomForest/<v1>&<p1>&<s1>&<r1>&<v2>&<p2>&<s2>&<r2>&<squadra1>&<squadra2>') # Route_1
api.add_resource(risultatoReteNeurale, '/risultatoReteNeurale/<v1>&<p1>&<s1>&<r1>&<v2>&<p2>&<s2>&<r2>&<squadra1>&<squadra2>') # Route_2
if __name__ == '__main__':
     app.run(port='5002')
