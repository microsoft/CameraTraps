#
# xmp_integration.py
#
# Tools for loading MegaDetector batch API results into XMP metadata, specifically
# for consumption in digiKam:
#
# https://cran.r-project.org/web/packages/camtrapR/vignettes/camtrapr2.html
#
    
#%% Imports and constants

import argparse
import tkinter
from tkinter import ttk, messagebox, filedialog

import inspect
import os
import sys
import json
import pyexiv2
import ntpath
import threading
import traceback

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from functools import partial

category_mapping = {'person': 'Human', 'animal': 'Animal', 'vehicle': 'Vehicle'}


#%% Class definitions

class xmp_gui:
    
    root = None
    textarea_min_threshold = None
    textarea_status = None
    textarea_remove_path = None
    textarea_rename_conf = None
    textarea_rename_cats = None
    num_threads = 1
    
class xmp_integration_options:
    
    # Folder where images are stored
    image_folder = None
    
    # .json file containing MegaDetector output
    input_file = None
    
    # String to remove from all path names, typically representing a 
    # prefix that was added during MegaDetector processing
    remove_path = None
    
    # Optionally *rename* (not copy) all images that have no detections
    # above [rename_conf] for the categories in rename_cats from x.jpg to
    # x.check.jpg
    rename_conf = None
    
    # Comma-deleted list of category names (or "all") to apply the rename_conf
    # behavior to.
    rename_cats = None
    
    # Minimum detection threshold (applies to all classes, defaults to None,
    # i.e. 0.0
    min_threshold = None
    num_threads = 1
    xmp_gui = None
    
    
#%% Functions
  
def write_status(options,s):
    
    if options.xmp_gui is None:
        return
    options.xmp_gui.textarea_status.configure(state="normal")
    options.xmp_gui.textarea_status.insert(tkinter.END, s + '\n')
    options.xmp_gui.textarea_status.configure(state="disabled")

    
n_images_processed = 0
    
def update_xmp_metadata(categories, options, rename_cats, n_images, image):
    """
    Update the XMP metadata for a single image
    """
    
    # Relative image path
    filename = ''
    
    # Absolute image path
    img_path = ''
    
    global n_images_processed
    
    try:
        
        filename = image['file']
        if options.remove_path != None and len(options.remove_path) > 0:
            filename = filename.replace(options.remove_path, '')
        img_path = os.path.join(options.image_folder, filename)
        assert os.path.isfile(img_path), 'Image {} not found'.format(img_path)
        
        # List of categories to write to XMP metadata
        image_categories = []
        
        # Categories with above-threshold detections present for
        # this image
        original_image_cats = []
        
        # Maximum confidence for each category
        original_image_cats_conf = {}
        
        for detection in image['detections']:
            
            cat = category_mapping[categories[detection['category']]]
            
            # Have we already added this to the list of categories to
            # write out to this image?
            if cat not in image_categories:
                
                # If we're supposed to compare to a threshold...
                if len(options.min_threshold) > 0 and \
                    options.min_threshold != None:
                    if float(detection['conf']) > float(options.min_threshold):
                        image_categories.append(cat)
                        original_image_cats.append(
                            categories[detection['category']])
                        
                # Else we treat *any* detection as valid...
                else:
                    image_categories.append(cat)
                    original_image_cats.append(categories[detection['category']])

            # Keep track of the highest-confidence detection for this class                
            if options.min_threshold != None and \
                len(options.min_threshold) > 0 and \
                    detection['conf'] > \
                        original_image_cats_conf.get(
                            categories[detection['category']], 0):
                            
                original_image_cats_conf[categories[detection['category']]] = \
                    detection['conf']
                    
        img = pyexiv2.Image(r'{0}'.format(img_path))
        img.modify_xmp({'Xmp.lr.hierarchicalSubject': image_categories})
        
        # If we're doing the rename/.check behavior...
        if not (options.rename_conf is None and options.rename_cats is None):
            
            matching_cats = set(rename_cats).intersection(set(original_image_cats))
            is_conf_low = False
            if options.min_threshold != None and len(options.min_threshold) > 0:
                for matching_cat in matching_cats:
                    if original_image_cats_conf[matching_cat] < float(options.rename_conf):
                        is_conf_low = True
            if options.min_threshold != None and \
                len(options.min_threshold) > 0 and \
                    len(image['detections']) == 0 or \
                (len(options.rename_conf) > 0 and \
                is_conf_low is True and \
                    len(matching_cats) > 0):
                        
                parent_folder = os.path.dirname(img_path)
                file_name = ntpath.basename(img_path)
                manual_file_name = file_name.split('.')[0]+'_check' + '.' + file_name.split('.')[1]
                os.rename(img_path, os.path.join(parent_folder, manual_file_name))
                
        if options.xmp_gui is not None:
            
            n_images_processed += 1
            percentage = round((n_images_processed)/n_images*100)
            options.xmp_gui.progress_bar['value'] = percentage
            options.xmp_gui.root.update_idletasks()
            options.xmp_gui.style.configure('text.Horizontal.Tprogress_bar',
                            text='{:g} %'.format(percentage))
                
    except Exception as e:
    
        s = 'Error processing image {}: {}'.format(filename,str(e))
        print(s)
        traceback.print_exc()
        write_status(options,s)
        
        if False:
            
            # Legacy code to rename files where XMP writing failed
            parent_folder = os.path.dirname(img_path)
            file_name = ntpath.basename(img_path)        
            failed_file_name = file_name.split('.')[0]+'_failed' + '.' + file_name.split('.')[1]
            os.rename(img_path, os.path.join(
                parent_folder, failed_file_name))


def process_input_data(options):
    """
    Main function to loop over images and modify XMP data
    """
    
    if options.xmp_gui is not None:
        
        if (options.image_folder is None) or (len(options.image_folder) == 0):
            tkinter.messagebox.showerror(title='Error', message='Image folder is not selected')
            sys.exit()
        if (options.input_file is None) or (len(options.input_file) == 0):
            tkinter.messagebox.showerror(
                title='Error', message='No MegaDetector .json file selected')
            sys.exit()
        options.remove_path = options.xmp_gui.textarea_remove_path.get()
        options.rename_conf = options.xmp_gui.textarea_rename_conf.get()
        options.rename_cats = options.xmp_gui.textarea_rename_cats.get()
        options.num_threads = options.xmp_gui.textarea_num_threads.get()
        options.min_threshold = options.xmp_gui.textarea_min_threshold.get()
            
    try:
        
        with open(options.input_file, 'r') as f:
            data = f.read()
    
        data = json.loads(data)
        categories = data['detection_categories']
    
        images = data['images']
        n_images = len(images)
        if not (options.rename_conf is None and options.rename_cats is None):
            rename_cats = options.rename_cats.split(",")
            if rename_cats[0] == 'all':
                rename_cats = list(category_mapping.keys())
        else:
            rename_cats = []
        if len(options.num_threads) > 0:
            num_threads = int(options.num_threads)
        else:
            num_threads = 1
        print(num_threads)
        if options.xmp_gui is None:
            func = partial(update_xmp_metadata, categories, options, rename_cats, n_images)
            with Pool(num_threads) as p:
                with tqdm(total=n_images) as pbar:
                    for i, _ in enumerate(p.imap_unordered(func, images)):
                        pbar.update()
        else:
            func = partial(update_xmp_metadata, categories, options, rename_cats, n_images)
            with ThreadPool(num_threads) as p:
                p.map(func, images)
            s = 'Successfully processed {} images'.format(n_images)
            print(s)
            write_status(options,s)
        
    except Exception as e:
        
        print('Error processing input data: {}'.format(str(e)))
        traceback.print_exc()
        if options.xmp_gui is not None:
            tkinter.messagebox.showerror(title='Error', 
                                         message='Make Sure you selected the proper image folder and JSON files')
        sys.exit()


def start_input_processing(options):
    
    t = threading.Thread(target=lambda: process_input_data(options))
    t.start()


def browse_folder(options,folder_path_var):
    
    filename = tkinter.filedialog.askdirectory()
    options.image_folder = r'{0}'.format(filename)
    folder_path_var.set(filename)


def browse_file(options,file_path_var):
    
    filename = tkinter.filedialog.askopenfilename()
    options.input_file = r'{0}'.format(filename)
    file_path_var.set(filename)


def create_gui(options):
    
    root = tkinter.Tk()
    root.resizable(False, False)
    root.configure(background='white')
    root.title('DigiKam Integration')
    
    group = tkinter.LabelFrame(root, padx=5, pady=5)
    group.configure(background = 'white')
    group.pack(padx=10, pady=10, fill='both', expand='yes')

    canvas = tkinter.Canvas(group, width = 800, height = 150)    
    canvas.configure(background = 'white')
    canvas.pack()      
    img1 = tkinter.PhotoImage(file='images/aiforearth.png')      
    canvas.create_image(0,0, anchor=tkinter.NW, image=img1)
    img2 = tkinter.PhotoImage(file='images/bg.png')      
    canvas.create_image(0,20, anchor=tkinter.NW, image=img2)

    frame = tkinter.Frame(root)
    frame.configure(background='white')
    frame.pack()
    
    l1 = tkinter.Label(frame, text='Folder containing images')
    l1.configure(background='white')
    l1.grid(row=0, column=0)
    
    folder_path_var = tkinter.StringVar()
    
    e1 = tkinter.Entry(frame, width=50, textvariable=folder_path_var, highlightthickness=1) 
    e1.configure(highlightbackground='grey', highlightcolor='grey')
    e1.grid(row=0, column=2)
    
    b1 = tkinter.Button(frame, text='Browse', fg='blue', command=lambda: browse_folder(options,folder_path_var))
    b1.grid(row=0, column=5, padx=10)
    
    l2 = tkinter.Label(frame, text='Path to MegaDetector output .json file') 
    l2.configure(background='white')
    l2.grid(row=1, column=0)
    
    file_path_var = tkinter.StringVar()
    
    e2 = tkinter.Entry(frame, width=50, textvariable=file_path_var, highlightthickness=1)
    e2.configure(highlightbackground='grey', highlightcolor='grey')    
    e2.grid(row=1, column=2)
    
    b2 = tkinter.Button(frame, text='Browse', fg='blue', command=lambda: browse_file(options,file_path_var))
    b2.grid(row=1, column=5, padx=10)

    l6 = tkinter.Label(frame, text='Minimum confidence to consider a category') 
    l6.configure(background='white')
    l6.grid(row=2, column=0)
    
    textarea_min_threshold = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_min_threshold.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_min_threshold.grid(row=2, column=2)
    
    l3 = tkinter.Label(frame, text='Prefix to remove from image paths (optional)') 
    l3.configure(background='white')
    l3.grid(row=3, column=0)
    
    textarea_remove_path = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_remove_path.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_remove_path.grid(row=3, column=2)

    l4 = tkinter.Label(frame, text='Confidence level to move images requires manual check (optional)') 
    l4.configure(background='white')
    l4.grid(row=4, column=0)

    textarea_rename_conf = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_rename_conf.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_rename_conf.grid(row=4, column=2)


    l5 = tkinter.Label(frame, text='Categories to check for the confidence (optional)') 
    l5.configure(background='white')
    l5.grid(row=5, column=0)

    textarea_rename_cats = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_rename_cats.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_rename_cats.grid(row=5, column=2)

    l6 = tkinter.Label(frame, text='Number of threads to run (optional)') 
    l6.configure(background='white')
    l6.grid(row=6, column=0)

    textarea_num_threads = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_num_threads.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_num_threads.grid(row=6, column=2)
    
    sb = tkinter.Button(frame, text='Submit', fg='black',
                command=lambda: start_input_processing(options), padx=10)
    sb.grid(row=7, column=2, padx=10, pady=10)

    style = tkinter.ttk.Style(root)
    style.layout('text.Horizontal.Tprogress_bar',
                [('Horizontal.progress_bar.trough',
                {'children': [('Horizontal.progress_bar.pbar',
                                {'side': 'left', 'sticky': 'ns'})],
                    'sticky': 'nswe'}),
                ('Horizontal.progress_bar.label', {'sticky': ''})])    
    style.configure('text.Horizontal.Tprogress_bar', text='0 %')

    progress_bar = tkinter.ttk.Progressbar(root, style='text.Horizontal.Tprogress_bar', length=700,
                                maximum=100, value=0, mode='determinate')
    progress_bar.pack(pady=10)

    group2 = tkinter.LabelFrame(root, text='Status', padx=5, pady=5)
    group2.pack(padx=10, pady=10, fill='both', expand='yes')

    textarea_status = tkinter.Text(group2, height=10, width=100)
    textarea_status.configure(state="disabled")
    textarea_status.pack()
    
    options.xmp_gui = xmp_gui()
    options.xmp_gui.root = root
    options.xmp_gui.textarea_min_threshold = textarea_min_threshold
    options.xmp_gui.textarea_remove_path = textarea_remove_path
    options.xmp_gui.textarea_rename_conf = textarea_rename_conf
    options.xmp_gui.textarea_rename_cats = textarea_rename_cats
    options.xmp_gui.textarea_num_threads = textarea_num_threads
    options.xmp_gui.textarea_status = textarea_status
    options.xmp_gui.progress_bar = progress_bar
    options.xmp_gui.style = style
    
    root.mainloop()
    
    
#%% Interactive/test driver

if False:
    
    #%%
    
    options = xmp_integration_options()
    options.input_file = r"C:\temp\demo_images\ssmini_xmp_test_orig\ssmini.mdv4.json"
    options.image_folder = r"C:\temp\demo_images\ssmini_xmp_test"
    options.remove_path = 'my_images/'
    process_input_data(options)
    
    
#%% Command-line driver

def args_to_object(args,obj):
    """
    Copy all fields from the argparse table "args" to the object "obj"
    """
    for n, v in inspect.getmembers(args):
        if not n.startswith('_'):
            setattr(obj, n, v)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help = 'Path to the MegaDetector .json file', default=None)
    parser.add_argument('--image_folder', help = 'Path to the folder containing images', default=None)
    parser.add_argument('--min_threshold', help = 'Minimum detection confidence that will be treated as a detection event', default=None)
    parser.add_argument('--remove_path', help = 'Prefix to remove from image paths in the .json file', default=None)
    parser.add_argument('--rename_conf', help = 'Below this confidence level, images will be renamed for manual check', default=None)
    parser.add_argument('--rename_cat', help = 'Category (or comma-delimited categories) to apply renaming behavior to', default=None)
    parser.add_argument('--num_threads', help = 'Number of threads to use for image processing', default=1)
    parser.add_argument('--gui', help = 'Run in GUI mode', action='store_true')
    
    options = xmp_integration_options()
    args = parser.parse_args()
    args_to_object(args,options)
    
    if options.gui:
        assert options.input_file is None, 'Command-line argument specified in GUI mode'
        assert options.image_folder is None, 'Command-line argument specified in GUI mode'
        assert options.min_threshold is None, 'Command-line argument specified in GUI mode'
        assert options.remove_path is None, 'Command-line argument specified in GUI mode'
        assert options.rename_conf is None, 'Command-line argument specified in GUI mode'
        assert options.rename_cat is None, 'Command-line argument specified in GUI mode'
        assert options.num_threads == 1, 'Command-line argument specified in GUI mode'
        create_gui(options)    
    else:
        process_input_data(options)


if __name__ == '__main__':
    
    main()
