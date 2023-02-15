from flask import Flask, jsonify
app=Flask(__name__)
@app.route("/")
def hello_world():
    return("<h1>hello world</h1>")
@app.route("/AboutUs")
def about_us():
    msg="<table border=1><tr><td>Name<td>Age<tr><td>Raj<td>12<tr><td>vishal<td>22"
    return msg

@app.route('/cube/<int:n>')
def cube(n):
    
    result =  {
                  "val": n*n*n,
                  "output": "success",
                } 
     
    return jsonify(result)


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5001)