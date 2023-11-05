import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, jsonify

app = Flask(__name__)


training = pd.read_csv('data/Training.csv')
testing = pd.read_csv('data/Testing.csv')
cols = training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)

scores = cross_val_score(clf, x_test, y_test, cv=3)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms_dict = {}

resp = []
question = ""
final = ""
count = 0
flagg = 0

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    global final
    sum=0
    for item in exp:
        if item in severityDictionary:
            sum = sum + severityDictionary[item]
        else:
            final += (f"Key '{item}' not found in severityDictionary!\n")
    if((sum*days)/(len(exp)+1)>13):
        final += ("\nYou should take the consultation from doctor. \n")
    else:
        
        final += ("\n It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('Data2/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)
            
def getSeverityDict():
    global severityDictionary
    with open('Data2/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Data2/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4],row[5],row[6]]}
            precautionDictionary.update(_prec)


# def getInfo():
#     print("-----------------------------------HealthCare ChatBot-----------------------------------")
#     print("\nEnter your name : ", end = " ")
#     name=input("")
#     print("Enter your age : ",end = "")
#     age = int(input())
#     print("Hello, ",name)

# getInfo()


getSeverityDict()
getDescription()
getprecautionDict()


def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def recurse(node, depth, tree_, feature_name, disease_input, symptoms_present, num_days):
    global flagg 
    global final, resp, question, count
    length = len(resp)
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]
        if name == disease_input:
            val = 1
        else:
            val = 0
        if val <= threshold:
            # Recurse left and return its result
            return recurse(tree_.children_left[node], depth + 1, tree_, feature_name, disease_input, symptoms_present, num_days)
        else:
            symptoms_present.append(name)
            # Recurse right and return its result
            return recurse(tree_.children_right[node], depth + 1, tree_, feature_name, disease_input, symptoms_present, num_days)
    else:
        present_disease = print_disease(tree_.value[node])
        red_cols = reduced_data.columns
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
        symptoms_exp = []
        index = 4
        for syms in list(symptoms_given):
            if length == index:
                symp = str(syms)
                symp = symp.replace("_", " ")
                return f"Do you have {symp}?"
            while True:
                if resp[index] == "yes" or resp[index] == "no":
                    break
                else:
                    del resp[index]
                    index -= 1
                    return "provide proper answers i.e. (yes/no) : "
            if resp[index] == "yes":
                symptoms_exp.append(syms)
            index += 1
        second_prediction = sec_predict(symptoms_exp)
        calc_condition(symptoms_exp, num_days)
        if present_disease[0] == second_prediction[0]:
            final += "\nYou may have " + present_disease[0] + "\n"
            final += description_list[present_disease[0]]
        else:
            final += "\nYou may have " + present_disease[0] + "or " + second_prediction[0] + "\n"
            final += description_list[present_disease[0]] + "\n"
            final += description_list[second_prediction[0]] + "\n"
        precution_list = precautionDictionary[present_disease[0]]
        final += "\n\n\nTake following measures : \n"
        for i, j in enumerate(precution_list):
            final += str(i + 1) + ")" + j + "\n"

        flagg = 1
        return final

    return "nothing"
               
def tree_to_code(tree, feature_names):
    global final
    global resp
    global question
    global count
    global flagg
    if flagg == 1:
        resp = []
        final = ""
        count = 0
        flagg = 0
    resp.append(question)
    length = len(resp)
    count = count + 1
    if count == 1:
        del resp[0]
        return "Welcome to doctor ROBO \nHow is your health?"
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        if count == 2:
            if resp[0] == "good" or resp[0] == "nice" or resp[0] == "ok":
                return "oh nice! If you have any type of symptom tell me"
            return "Enter the symptom you are experiencing"
        disease_input = resp[1]

        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            temp = "searches related to input: \n"
            for num, it in enumerate(cnf_dis):
                temp += str(num) + ")" + it + "\n"
            if num != 0:
                temp += f"Select the one you meant (0 - {num})"
                if count == 3:
                    temp = re.sub('_', ' ', temp)
                    return (temp)
                conf_inp = int(resp[2])
            else:
                if count == 3:
                    count += 1
                    resp.append("NA")
                conf_inp = 0

            disease_input=cnf_dis[conf_inp]
            break

        else:
            count -= 1
            del resp[1]
            return "Enter valid symptom."
        
    while True:
        try:
            if count == 4:
                return ("Okay. From how many days ? : ")
            num_days = int(resp[3])
            break
        except:
            count -= 1
            del resp[3]
            return "Enter valid input." 
          
    ret = recurse(0, 1, tree_, feature_name, disease_input, symptoms_present, num_days)
    return ret


def generate(respo):
    global question 
    question = respo
    print("question : ", question)
    rep = tree_to_code(clf,cols)
    print("bot response : ", rep)
    return rep

@app.route('/', methods=['POST'])
def respond_to_input():
    data = request.get_json()
    user_message = data.get('question')

    if user_message:
        if isinstance(user_message, int):
            user_message = str(user_message)
        response = generate(user_message)
    else:
        response = "Please provide a message."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True) 