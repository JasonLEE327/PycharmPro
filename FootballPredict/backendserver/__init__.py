#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: li
'''

#init
from flask import Flask
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

from backendserver.controller import controller

