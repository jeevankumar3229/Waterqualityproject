from flask import Flask, render_template, request, jsonify
from joblib import load
from flask import Flask, render_template, request, redirect, session
import mysql.connector
import numpy as np
import pandas as pd
import serial

app = Flask(__name__, template_folder='new')
app.secret_key = "secret_key"

model = load("Models\extratree_water.joblib")
model_lr= load("Models\extratree_water.joblib")
df = pd.read_csv('CSV/new_data.csv')
print(df.shape)

df.drop(df[df['Potability'] == 0].index, inplace=True)

print(df.shape)
ph_mean = df['ph'].mean()
solids_mean = df['Solids'].mean()
turbidity_mean = df['Turbidity'].mean
temperature_mean = df['Temperature'].mean()
conductivity_mean = df['Conductivity'].mean()
hardness_mean = df['Hardness'].mean()
chloramines_mean = df['Chloramines'].mean()
sulfate_mean = df['Sulfate'].mean()
organiccompound_mean = df['Organic_Carbon'].mean()

print(ph_mean)
print(hardness_mean)
print(solids_mean)
print(chloramines_mean)
print(sulfate_mean)
print(conductivity_mean)
print(organiccompound_mean)
print(temperature_mean)
print(turbidity_mean)


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="water_quality1"
)



@app.route("/")
def home():
    if 'username' in session:
        return redirect("/predict")
    else:
        return render_template("home.html")

@app.route("/home1")
def home1():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/loginp")
def loginp():
    return render_template("login.html")


@app.route("/resultp")
def resultp():
    return render_template("result.html")

@app.route("/predictp")
def predictp():
        ser = serial.Serial('COM3', 9600)  
        data = ser.readline().decode().rstrip()  
        values = data.split(' ')  
        for value in values:
            if value.startswith("Turbidity:"):
                turb =  value.split(":")[1]
            elif value.startswith("TDS:"):
                tds = value.split(":")[1]
            elif value.startswith("Temperature"):
                temp = value.split(":")[1]
            elif value.startswith("pH"):
                phval = value.split(":")[1]
            elif value.startswith("Conductivity"):
                cond = value.split(":")[1]
        prediction1 = model.predict(np.array([[phval,tds,turb,temp,cond,hardness_mean,chloramines_mean,sulfate_mean,organiccompound_mean]]));
      #  prediction_lr1 = model_lr.predict_proba(np.array([[phval,tds,turb,temp,cond,hardness_mean,chloramines_mean,sulfate_mean,organiccompound_mean]]));
      #  prediction_lr_q1 = (1 - prediction_lr1[0][0]) * 100
        if prediction1 ==0:
            predict = "Bad"
        else:
            predict = "Good"
            # mycursor = mydb.cursor()
            # query = "INSERT INTO predictions (PH, Hardness, Solids, Chloramines, Sulfate, Conductivity, OrganicCarbon, Temperature, Turbidity, result) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            # values = (phval,hardness_mean,tds,chloramines_mean,sulfate_mean,cond,organiccompound_mean,temp,turb,prediction1,prediction_lr_q1)
            # mycursor.execute(query, values)
            # mydb.commit() 
        return render_template("predict.html",ph = phval,solids=tds,chloramines=chloramines_mean,hardness=hardness_mean,sulfate=sulfate_mean,conductivity=cond,organic_carbon=organiccompound_mean,temperature=temp,turbidity=turb,result=predict)

@app.route("/reportp")
def reportp():
    cur = mydb.cursor()
    cur.execute('SELECT ph,Solids,Turbidity,Temperature,Conductivity,Hardness,Chloramines,Sulfate,Organic_Carbon,Result,Quality FROM report')
    data = cur.fetchall()

    return render_template('report.html', data=data)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        name = request.form['name']
        password = request.form['password']

        mycursor = mydb.cursor()
        query = "SELECT * FROM admin_detail WHERE username=%s AND password=%s"
        values = (username, password)
        mycursor.execute(query, values)
        result = mycursor.fetchone()

        if result:
            return render_template("signup.html", error="Account already exists")
        else:
            cursor = mydb.cursor()
            query = "INSERT INTO admin_detail (username, name, password) VALUES (%s, %s, %s)"
            values = (username, name, password)
            cursor.execute(query, values)
            mydb.commit()

            return render_template("login.html")
    else:
        return render_template("signup.html")  

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        mycursor = mydb.cursor()
        query = "SELECT * FROM admin_detail WHERE username=%s AND password=%s"
        values = (username, password)
        mycursor.execute(query, values)
        result = mycursor.fetchone()

        if result:
            session['username'] = username
            return redirect("/predict")
        else:
            return render_template("login.html", error="Invalid username or password ")
    else:
        return render_template("login.html")
    
@app.route("/predict", methods=["GET", "POST"])
def prediction():
    if 'username' in session:
        if request.method == "POST":
            ph = float(request.form['PH'])
            Hardness = float(request.form['Hardness'])
            Solids = float(request.form['Solids'])
            Chloramines = float(request.form['Chloramines'])
            Sulfate = float(request.form['Sulfate'])
            Conductivity = float(request.form['Conductivity'])
            Organic_Carbon = float(request.form['OrganicCarbon'])
            Temperature = float(request.form['Temperature'])
            Turbidity = float(request.form['Turbidity'])

            prediction = model.predict(np.array([[ph,Solids,Turbidity,Temperature,Conductivity,Hardness,Chloramines,Sulfate,Organic_Carbon]]));
            prediction_lr = model_lr.predict_proba(np.array([[ph,Solids,Turbidity,Temperature,Conductivity,Hardness,Chloramines,Sulfate,Organic_Carbon]]));
            prediction_lr_q = (1 - prediction_lr[0][0]) * 100
            # mycursor = mydb.cursor()
            # query = "INSERT INTO predictions (username, PH, Hardness, Solids, Chloramines, Sulfate, Conductivity, OrganicCarbon, Trihalomethanes, Turbidity, result) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            # values = (session['username'], PH, Hardness, Solids, Chloramines, Sulfate, Conductivity, OrganicCarbon, Trihalomethanes, Turbidity, str(prediction[0]))
            # mycursor.execute(query, values)
            # mydb.commit() 
            if prediction == 0:
                output = " BAD"
            else:
                output ="GOOD"
            return render_template('result.html', prediction='Predicted Quality: {}'.format(output))
        else:
            return render_template("result.html")
    else:
        return redirect("/login")

@app.route("/logout")
def logout():

    session.clear()
    session.pop('username', None)
    return redirect("/login")
@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response

if __name__ == "__main__":
    app.run(debug=True)
