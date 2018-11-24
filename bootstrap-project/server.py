from flask import Flask, request, render_template   
import app.hello as hello

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
# renders the view
def index():
    if request.method == "POST":
        name = request.form["name"]
        return name + " Hello"
    return render_template("index.html")

@app.route("/hello", methods=['GET'])
def test():
    hello.hello_test() # this prints out on to command prompt
    return render_template("index.html")

@app.route("/translate", methods=['GET'])
def test2():
    hello.test() # this prints out on to command prompt
    return render_template("index.html")

@app.route("/get_histo", methods=['GET'])
def test3():
    hello.test() # this prints out on to command prompt
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug = True)