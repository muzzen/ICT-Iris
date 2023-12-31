from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['GET','POST'])
def result():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        model = pickle.load(open('model.pkl','rb'))
    
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
   
        return render_template('result.html', 
                               species= result[0])
    else:
        return render_template('result.html')

if __name__ == '__main__':
    app.run()
