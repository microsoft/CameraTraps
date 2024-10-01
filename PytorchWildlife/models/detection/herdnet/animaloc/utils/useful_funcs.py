__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import os 
import errno
from datetime import date, datetime

def mkdir(path):
    ''' To make a directory from a path '''
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def current_date():
    ''' To get current date in YYYYMMDD format '''
    today = date.today().strftime('%Y%m%d')
    return today

def get_date_time():
    ''' To get current date and time in "d/m/Y H:M:S" format '''
    today = date.today().strftime('%d/%m/%Y')
    now = datetime.now().strftime('%H:%M:%S')
    return today , now

def vdir(obj):
    ''' To filter out 'special methods' of an object '''
    return [m for m in dir(obj) if not m.startswith('__')]