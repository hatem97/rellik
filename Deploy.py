from flask  import flask
app = Flask(_name_)

@app.route('/')
def index():
return '<h1>Depoyed to heroku!!</h1>'