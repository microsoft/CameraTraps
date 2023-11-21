#
# eMammal_helpers.py
#
# Support functions for processing eMammal metadata
#

#%% Constants and imports

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import operator
from datetime import datetime


#%% Support functions

def clean_species_name(common_name):
    """
    Converts various forms of "human" to the token "human", and various forms
    of "empty" to the token "empty"
    """
    
    _people_tags = {
        'Bicycle',
        'Calibration Photos',
        'Camera Trapper',
        'camera trappper',
        'camera  trapper',
        'Homo sapien',
        'Homo sapiens',
        'Human, non staff',
        'Human, non-staff',
        'camera trappe',
        'Human non-staff',
        'Setup Pickup',
        'Vehicle'
    }
    PEOPLE_TAGS = {x.lower() for x in _people_tags}

    _no_animal_tags = {'No Animal', 'no  animal', 'Time Lapse', 'Camera Misfire', 'False trigger', 'Blank'}
    NO_ANIMAL_TAGS = {x.lower() for x in _no_animal_tags}

    common_name = common_name.lower().strip()
    if common_name in PEOPLE_TAGS:
        return 'human'

    if common_name in NO_ANIMAL_TAGS:
        return 'empty'

    return common_name


def clean_frame_number(img_frame):
    
    # pad to a total of 3 digits if < 1000, or 4 digits otherwise
    # img_frame is a string from the xml tree
    length = len(img_frame)

    assert length > 0
    assert length < 5

    # length 4 frame order is returned as is, others are left padded to be 3 digit long
    # we need to make sure img_frame has length 3 when it's < 1000 so we can match it to the iMerit labels
    if length == 1:
        return '00' + img_frame
    elif length == 2:
        return '0' + img_frame
    else:  # for '100' and '1000'
        return img_frame


def clean_frame_number_4_digit(img_frame):
    
    # pad to a total of 4 digits
    # img_frame is a string from the xml tree
    length = len(img_frame)

    assert length > 0
    assert length < 5

    # length 4 frame order is returned as is, others are left padded to be 3 digit long
    # we need to make sure img_frame has length 3 when it's < 1000 so we can match it to the iMerit labels
    if length == 1:
        return '000' + img_frame
    elif length == 2:
        return '00' + img_frame
    elif length == 3:
        return '0' + img_frame
    else:  # for'1000'
        return img_frame


def get_img_size(img_path):
    """
    There are ways to size the image without loading it into memory by reading its headers
    (https://github.com/scardine/image_size), but seems less reliable.

    Returns (-1, -1) if PIL could not open the image
    """

    try:
        im = Image.open(img_path)
        width, height = im.size
    except:
        return (-1, -1)
    return (width, height)


def get_total_from_distribution(d):
    
    total = 0
    for key, count in d.items():
        total += int(key) * count
    return total


def sort_dict_val_desc(d, percent=False):    
    """ Sort a dictionary by the values in descending order. Returns a list of tuples. """
    
    sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)

    if percent:
        with_percent = []
        total = sum([t[1] for t in sorted_d])
        for k, v in sorted_d:
            p = '{:.1f}%'.format(100 * float(v) / total)
            with_percent.append((k, v, p))
        return with_percent

    return sorted_d


def plot_distribution(d, title='', top=15):
    
    if top is None or top > len(d):
        top = len(d)

    sorted_d = sort_dict_val_desc(d)

    top_d = sorted_d[:top]
    x = [t[0] for t in top_d]
    y = [t[1] for t in top_d]

    # others column
    others_d = sorted_d[top:]
    others_sum = sum([t[1] for t in others_d])
    x.append('others')
    y.append(others_sum)

    plt.bar(range(len(x)), y, align='center', facecolor='#57BC90', edgecolor=None)
    plt.xticks(range(len(x)), x, rotation=90)
    plt.title(title)
    plt.show()


def plot_histogram(l, title='', max_val=None, bins='auto'):
    
    if max_val:
        l = [x for x in l if x < max_val]

    plt.hist(l, bins=bins, facecolor='#57BC90', edgecolor=None)
    plt.title(title)
    plt.show()


def draw_bboxes(image, bboxes, classes, thickness=4, show_label=False):    
    """
    Draw bounding boxes on top of an image
    Args:
        image : Path to image or a loaded PIL image
        bboxes: A list of bboxes to draw on the image, each bbox is [top left x, top left y, width, height] in relative coordinates
        classes: A list of classes corresponding to the bboxes
        thickness: Thickness of the line to draw, minimum is 1
    Outputs:
        Image object with the bboxes and class labels annotated
    """
    if type(image) is str:
        img = Image.open(image)
    else:
        img = image.copy()

    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    for i in range(len(bboxes)):
        x_rel, y_rel, w_rel, h_rel = bboxes[i]
        x = x_rel * img_width  # x and y are the top left
        y = y_rel * img_height
        w = w_rel * img_width
        h = h_rel * img_height

        if show_label:
            c = classes[i]
            draw.text((x + 15, y + 15), str(c), fill=(255, 0, 0, 255))

        for j in range(thickness):
            draw.rectangle(((x + j, y + j), (x + w + j, y + h + j)), outline='red')

    return img


def is_daytime(date_time):
    """ 
    Returns True if daytime as determined by the input timestamp, a rough 
    decision based on two seasons
    """
    
    # summer day hours: 6am - 7pm
    # others day hours: 7am - 6pm

    is_summer = True if date_time.month in [5, 6, 7, 8, 9] else False
    if is_summer:
        if date_time.hour >= 6 and date_time.hour <= 19:
            return True
        else:
            return False
    else:
        if date_time.hour >= 7 and date_time.hour <= 18:
            return True
        else:
            return False


def parse_timestamp(time_str):    
    """
    There are three datetime string formats in eMammal, and some have an empty field.
    Args:
        time_str: text in the tag ImageDateTime

    Returns:
        datetime object, error (None if no error)
    """

    if time_str == '' or time_str is None:
        return '', 'empty or None'
    try:
        res = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        return res, None
    except Exception:
        try:
            res = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return res, None
        except Exception:
            try:
                res = datetime.strptime(time_str, '%m/%d/%Y %H:%M')
                return res, None
            except:
                print('WARNING, time_str cannot be parsed {}.'.format(time_str))
                return time_str, 'cannot be parsed {}'.format(time_str)  # return original string
