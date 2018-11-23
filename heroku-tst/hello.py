#example from http://octomaton.blogspot.com/2014/07/hello-world-on-heroku-with-python.html
from flask import Flask

app = Flask(__name__)

@app.route('/')
def source():
 html = 'Hello World!'

 #return html
 return render_template('index.html')

 if __name__ == "__main__":
 	app.debug = True
 	app.run()