import requests
import os
import re
from flask import Flask, Response, jsonify
from flask import request, session

def save_return_path():
    path =  str(request.url)
    #if(path == "/"):
    #    path = "index"
    session['path'] = path

def is_logged_in():
    if 'logged_in' in session:
        print('logged in')
        return True
    else:
        return False

def redirect_to_login():
    save_return_path()
    root_url = get_root_url(request.url)
    login_url = root_url + "/login"
    resp = Response(status=307)
    resp.headers['location'] = login_url
    return resp

def get_root_url(url):
    
    p = '(?:http.*://)?(?P<host>[^:/ ]+).?(?P<port>[0-9]*).*'
    
    m = re.search(p,url)
    host = m.group('host')
    port = m.group('port')
    
    if(port == ""):
        return 'http://' + m.group('host') 
    return 'http://' + m.group('host') + ":" + port
