from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

classifier = pickle.load(open('classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    

    predicted_species_name = classifier.predict(user_input)[0]

    return render_template('result.html', 
                           species=predicted_species_name)

if __name__ == '__main__':
    app.run(port=8000)