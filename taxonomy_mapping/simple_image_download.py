#
# simple_image_download.py
#
# Cloned from:
#
# https://github.com/RiddlerQ/simple_image_download
#
# Slighty modified to take an output directory.
#

import os
import urllib
import requests
from urllib.parse import quote

class simple_image_download:
    def __init__(self):
        pass

    def urls(self, keywords, limit):
        keyword_to_search = [str(item).strip() for item in keywords.split(',')]
        i = 0
        links = []
        while i < len(keyword_to_search):
            url = 'https://www.google.com/search?q=' + quote(
                keyword_to_search[i].encode(
                    'utf-8')) + '&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'
            raw_html = self._download_page(url)

            end_object = -1;

            j = 0
            while j < limit:
                while (True):
                    try:
                        new_line = raw_html.find('"https://', end_object + 1)
                        end_object = raw_html.find('"', new_line + 1)

                        buffor = raw_html.find('\\', new_line + 1, end_object)
                        if buffor != -1:
                            object_raw = (raw_html[new_line + 1:buffor])
                        else:
                            object_raw = (raw_html[new_line + 1:end_object])

                        if '.jpg' in object_raw or 'png' in object_raw or '.ico' in object_raw or '.gif' in object_raw or '.jpeg' in object_raw:
                            break

                    except Exception as e:
                        print(e)
                        break


                try:
                    r = requests.get(object_raw, allow_redirects=True)
                    if('html' not in str(r.content)):
                        links.append(object_raw)
                    else:
                        j -= 1
                except Exception as e:
                    print(e)
                    j -= 1
                j += 1

            i += 1
            
        return(links)


    def download(self, keywords, output_directory, limit=50):
        
        image_paths = []
        
        keyword_to_search = [str(item).strip() for item in keywords.split(',')]
        os.makedirs(output_directory,exist_ok=True)
        i = 0

        while i < len(keyword_to_search):
            url = 'https://www.google.com/search?q=' + quote(
                keyword_to_search[i].encode('utf-8')) + '&biw=1536&bih=674&tbm=isch&sxsrf=ACYBGNSXXpS6YmAKUiLKKBs6xWb4uUY5gA:1581168823770&source=lnms&sa=X&ved=0ahUKEwioj8jwiMLnAhW9AhAIHbXTBMMQ_AUI3QUoAQ'
            raw_html = self._download_page(url)

            end_object = -1;

            j = 0
            while j < limit:
                while (True):
                    try:
                        new_line = raw_html.find('"https://', end_object + 1)
                        end_object = raw_html.find('"', new_line + 1)

                        buffor = raw_html.find('\\', new_line + 1, end_object)
                        if buffor != -1:
                            object_raw = (raw_html[new_line+1:buffor])
                        else:
                            object_raw = (raw_html[new_line+1:end_object])

                        if '.jpg' in object_raw or 'png' in object_raw or '.jpeg' in object_raw: 
                            # or '.ico' in object_raw or '.gif' in object_raw:
                            break

                    except Exception as e:
                        print(e)
                        break
                
                # Todo: we have no evidence these are jpegs
                filename = str(keyword_to_search[i]) + "_" + '{:03}'.format(j) + ".jpg"

                try:
                    image_path = os.path.join(output_directory, filename)
                    r = requests.get(object_raw, allow_redirects=True)
                    if('html' not in str(r.content)):
                        open(image_path, 'wb').write(r.content)
                        image_paths.append(image_path)
                    else:
                        j -= 1                    
                except Exception as e:
                    print(e)
                    j -= 1
                j += 1

            i += 1

        return image_paths

    def _download_page(self,url):

        try:
            headers = {}
            headers['User-Agent'] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"
            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req)
            respData = str(resp.read())
            return respData

        except Exception as e:
            print(e)
            exit(0)
