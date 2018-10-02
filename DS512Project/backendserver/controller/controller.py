#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: li
'''
from backendserver import app
from flask import request,render_template,flash,abort,url_for,redirect,session,Flask,g
import find_position
from backendserver.service import Match_Predict

# from flask_nav import Nav
# from flask_nav.elements import *

'''
nav=Nav()
nav.register_element('top',Navbar(u'512 PROJECT',
                                    View(u'GET_PREDICT','get_predict'),
                                    View(u'GET_POSITION','to_position_predict'),
                                    #Subgroup(u'项目',
                                    #         View(u'项目一','about'),
                                    #         Separator(),
                                    #         View(u'项目二', 'service'),
                                    #),
))
nav.init_app(app)
'''

data = None


@app.route('/', methods=['GET', 'POST'])
def start():
    # data =
    return render_template('index.html')


@app.route('/index', methods=['GET', 'POST'])
def to_index():
    # data =
    return render_template('index.html')


@app.route('/MatchPredict', methods=['GET', 'POST'])
def to_match_predict():
    return render_template('MatchPredict.html')


@app.route('/test', methods=['GET', 'POST'])
def to_test():
    return render_template('test.html')


@app.route('/backend/get_predict', methods=['POST'])
def get_predict():
    home_away = request.form.get('home_away', 0.0473)
    # term1 = request.form.get('term1', 'term1 default value')
    term1 = request.form.getlist('term1[]')
    term2 = request.form.getlist('term2[]')
    # term2 = request.form.get('term2', 'term2 default value')
    formation1 = request.form.getlist('formation1[]')
    formation2 = request.form.getlist('formation2[]')
    t1 = []
    for i in term1:
        t1.append(int(i))
    t2 = []
    for i in term2:
        t2.append(int(i))
    f1 = []
    for i in formation1:
        f1.append(int(i))
    f2 = []
    for i in formation2:
        f2.append(int(i))
    predict = Match_Predict.Predict(t1, t2, f1, f2, home_away)
    r = predict.Predict()

    print home_away, t1, t2, f1, f2
    print r
    result = "{\"t1win\" :%0.4f, \"t2win\" :%0.4f,\"draw\" :%0.4f,\"expect1\" :%0.4f,\"expect2\" :%0.4f}"

    # "t1win", t1win, "t2win", t2win, "draw", t1t2even
    return result % (r[0], r[1], r[2], r[3], r[4])


@app.route('/PositionPredict',methods=['GET','POST'])
def to_position_predict():
    attributes = (['Aggression', 'Crossing', 'Curve', 'Dribbling'],
                  ['Finishing', 'Free_kick_accuracy', 'Heading_accuracy', 'Long_shots'],
                  ['Penalties', 'Shot_power', 'Volleys', 'Short_passing'],
                  ['Long_passing', 'Interceptions', 'Marking', 'Sliding_tackle'],
                  ['Standing_tackle', 'Strength', 'Vision', 'Acceleration'],
                  ['Agility', 'Reactions', 'Stamina', 'Balance'],
                  ['Ball_control', 'Composure', 'Jumping', 'Sprint_speed'],
                  ['Positioning'])
    return render_template('position.html', attributes=attributes)

@app.route('/backend/get_position', methods=['GET','POST'])
def get_position():
    a = request.form.getlist('attributes[]')

    for i in a:
        try:
            int(i)
        except:
            return
        if int(i) > 100 or int(i) < 0:
            return

    attributes = []
    for item in a:
        print item
        if item == '':
            item = 1
        attributes.append(int(item))
    # print attributes
    fp = find_position.FindPosition(file='CompleteDataset.csv', attributes=attributes)
    [best_pos, prob] = fp.predict()

    return best_pos


