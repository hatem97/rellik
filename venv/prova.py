# -*- coding: utf-8 -*-
#from __future__ import unicode_literals
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class prova(Resource):
    def get(self,*attr):
        print ("Ciao: %s" % attr)
        #print ("ciao "+str(%attr))
        return attr
api.add_resource(prova, "/prova") # Route_1
if __name__ == '__main__':
     app.run(port='5002')