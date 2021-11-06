#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Heart_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    
    las_features=[]
    for index,value in enumerate(int_features):
        if index==0:
           if value==0:
              las_features.append(1)
              las_features.append(0)
           else:
              las_features.append(0)
              las_features.append(1)
        elif index==1:
           if value==0:
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
           elif value==1:
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
           elif value==2:
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
           else :
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
        elif index==2:
           if value==0:
              las_features.append(1)
              las_features.append(0)
           else:
              las_features.append(0)
              las_features.append(1)
        elif index==3:
           if value==0:
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
           elif value==1:
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
           else:
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
        elif index==4:
           if value==0:
              las_features.append(1)
              las_features.append(0)
           else:
              las_features.append(0)
              las_features.append(1)
        elif index==5:
           if value==0:
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
           elif value==1:
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
           else:
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
        elif index==6:
           if value==0:
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
           elif value==1:
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
           elif value==2:
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
           elif value==3:
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
           else :
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
        elif index==7:
           if value==0:
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
           elif value==1:
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
              las_features.append(0)
           elif value==2:
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
              las_features.append(0)
           else :
              las_features.append(0)
              las_features.append(0)
              las_features.append(0)
              las_features.append(1)
        else:
              las_features.append(value)
    

            
    finall_features = [np.array(las_features)]
    prediction = model.predict(finall_features)

    output = prediction[0]  
    msg=""
    
    if output==0:
       msg="He/She haven't heart disease" 
    else:
       msg="He/She have heart disease !"
    return render_template('index.html', prediction_text=msg)


if __name__ == "__main__":
    app.run(debug=True)

