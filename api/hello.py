from flask import Flask

app = Flask(__name__)

@app.route("/mohit/<name>")
def hello_world(name):
    return "<p>Hello, World!</p>" + name
@app.route("/<a>/<b>")
def submission(n1,n2):
    return int(n1)+ int(n2)