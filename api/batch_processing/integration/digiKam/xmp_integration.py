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
import inspect
import os
import sys
import json
import pyexiv2
import ntpath
import threading
import traceback

from tkinter import ttk, messagebox, filedialog

from tqdm import tqdm

# Debug-only variable to rename images considered empty
confidence_threshold = 0.9


#%% Class definitions

class xmp_gui:
    
    root = None
    textarea_status = None
    textarea_removepath = None
    
class xmp_integration_options:
    
    image_folder = None
    input_file = None
    remove_path = None
    xmp_gui = None
    
    
#%% Functions
  
def write_status(options,s):
    
    if options.xmp_gui is None:
        return
    options.xmp_gui.textarea_status.configure(state="normal")
    options.xmp_gui.textarea_status.insert(tkinter.END, s + '\n')
    options.xmp_gui.textarea_status.configure(state="disabled")
    
    
def update_xmp_metadata(image, categories, options):
    """
    Update the XMP metadata for a single image
    """
    filename = ''
    img_path = ''
    
    try:
        
        filename = image['file']
        if options.remove_path != None and len(options.remove_path) > 0:
            filename = filename.replace(options.remove_path, '')
        img_path = os.path.join(options.image_folder, filename)
        assert os.path.isfile(img_path), 'Image {} not found'.format(img_path)
        image_categories = []
        for detection in image['detections']:
            cat = categories[detection['category']]
            if cat not in image_categories:
                image_categories.append(cat)
        img = pyexiv2.Image(r'{0}'.format(img_path))
        img.modify_xmp({'Xmp.lr.hierarchicalSubject': image_categories})
                       
    except Exception as e:
    
        s = 'Error processing image {}: {}'.format(filename,str(e))
        print(s)
        traceback.print_exc()
        write_status(options,s)
        

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
        options.remove_path = options.xmp_gui.textarea_removepath.get()
            
    try:
        
        with open(options.input_file, 'r') as f:
            data = f.read()
    
        data = json.loads(data)
        categories = data['detection_categories']
    
        images = data['images']
        n_images = len(images)
        if options.xmp_gui is None:
            for index, image in tqdm(enumerate(images),total=n_images):
                update_xmp_metadata(image, categories, options)
        else:
            
            for index, image in enumerate(images):
                percentage = round((index+1)/n_images*100)
                options.xmp_gui.progress_bar['value'] = percentage
                options.xmp_gui.root.update_idletasks()
                options.xmp_gui.style.configure('text.Horizontal.Tprogress_bar',
                                text='{:g} %'.format(percentage))
                update_xmp_metadata(image, categories, options)
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
    
    l3 = tkinter.Label(frame, text='Prefix to remove from image paths (optional)') 
    l3.configure(background='white')
    l3.grid(row=2, column=0)
    
    textarea_removepath = tkinter.Entry(frame, width=50, highlightthickness=1)
    textarea_removepath.configure(highlightbackground='grey', highlightcolor='grey')
    textarea_removepath.grid(row=2, column=2)

    sb = tkinter.Button(frame, text='Submit', fg='black',
                command=lambda: start_input_processing(options), padx=10)
    sb.grid(row=3, column=2, padx=10, pady=10)

    style = tkinter.ttk.Style(root)
    style.layout('text.Horizontal.Tprogress_bar',
                [('Horizontal.progress_bar.trough',
                {'children': [('Horizontal.progress_bar.pbar',
                                {'side': 'left', 'sticky': 'ns'})],
                    'sticky': 'nswe'}),
                ('Horizontal.progress_bar.label', {'sticky': ''})])    
    style.configure('text.Horizontal.Tprogress_bar', text='0 %')

    progress_bar = tkinter.ttk.Progressbar(root, style='text.Horizontal.Tprogress_bar', length=700,
                                maximum=100, value=0)
    progress_bar.pack(pady=10)

    group2 = tkinter.LabelFrame(root, text='Status', padx=5, pady=5)
    group2.pack(padx=10, pady=10, fill='both', expand='yes')

    textarea_status = tkinter.Text(group2, height=10, width=100)
    textarea_status.configure(state="disabled")
    textarea_status.pack()
    
    options.xmp_gui = xmp_gui()
    options.xmp_gui.root = root
    options.xmp_gui.textarea_removepath = textarea_removepath
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
    parser.add_argument('--remove_path', help = 'Prefix to remove from image paths in the .json file (optional)', default=None)
    parser.add_argument('--gui', help = 'Run in GUI mode', action='store_true')
    
    options = xmp_integration_options()
    args = parser.parse_args()
    args_to_object(args,options)
    
    if options.gui:
        assert options.input_file is None, 'Command-line argument specified in GUI mode'
        assert options.image_folder is None, 'Command-line argument specified in GUI mode'
        assert options.remove_path is None, 'Command-line argument specified in GUI mode'
        create_gui(options)    
    else:
        process_input_data(options)


if __name__ == '__main__':
    
    main()
