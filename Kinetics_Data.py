#!/usr/bin/env python
# coding: utf-8

# In[10]:





# In[1]:


import json
import pandas as pd
import os
from tqdm import tqdm

name = "test"

path = "K400/"

dir = name + "/"

# Opening JSON file
f = open("D:/ML_notebooks/kinetics400/" + name +".json")
js = json.load(f)
df = pd.read_csv('D:/ML_notebooks/kinetics400/'+name+'.csv')
data = df

labels = data["label"].drop_duplicates()
len(labels)


# In[2]:


#Create Directory

for i in labels:
    mypath = path + dir + i
    if not os.path.isdir(mypath):
       os.makedirs(mypath)


# In[25]:


import imageio
import cv2
import os
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
        

def get_frames(title,mypath,num):
    
    #print(title)
    
    vidcap = cv2.VideoCapture(title)
    
    success,image = vidcap.read()
    
    #print(success)
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))    # added this line 
        success,image = vidcap.read()
        #print ('Read a new frame: ', success)
        try:
            cv2.imwrite( mypath + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        except Exception:
            pass
        count = count + 1
        if(count == num):
            success = False
    
    if (count < num ):
        
        try:
            remove(mypath)
        except Exception:
            pass
        
            
#get_frames("World Beer Drinking Record 1975 The Whippet! An Aussie Beer Lover---6bJUbfpnQ.mkv","K400/test/drinking beer/--6bJUbfpnQ",10)


# In[123]:





# In[ ]:





# In[ ]:


from __future__ import unicode_literals
import youtube_dl

def check(path):
    
    if len(os.listdir(path)) == 0:
        return 0
    else:    
        return 1

class FilenameCollectorPP(youtube_dl.postprocessor.common.PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information['filepath'])
        return [], information

    
for i in tqdm(labels):
    list = data.loc[data['label'] == i ]
    ids = list['youtube_id']
    
    
    for j in tqdm(ids):
        mypath = path + dir + i + '/' + j
        if not os.path.isdir(mypath):
           os.makedirs(mypath)

        
        mypath = path + dir + i + '/' + j
        
        if( check(mypath) == 1 ) :
            continue
        
        ur = js[j]['url']
        
        ydl_opts = {
                'format': 'bestvideo[ext=mp4][height<=144]'
            }

        video_title = ''
        filename_collector = FilenameCollectorPP()
        
        my_youtube_dl = youtube_dl.YoutubeDL(ydl_opts)
        
        with my_youtube_dl as ydl:
            
            inf = 100
            
            my_youtube_dl.add_post_processor(filename_collector)
            
            try:
                info_dict = my_youtube_dl.extract_info(ur, download=False)
                inf = info_dict['duration']
            except Exception:
                pass
            
            if (inf < 30 ):
            
                try:
                    info_dict = my_youtube_dl.extract_info(ur, download=True)
                    video_title = info_dict.get('title', None)
            
                    title = filename_collector.filenames[0]
                
                    print(filename_collector.filenames[0])
                
                except Exception:
                
                    pass
                
                try:
                    get_frames(title,mypath,10)
                
                except Exception:
                
                    pass
            
                try:
                    remove(title)
                except Exception:
                    pass
            
            try:
                
                if( check(mypath) == 0 ) :
                    try:
                        remove(mypath)
                    except Exception:
                        pass
            except Exception:
                
                pass


# In[101]:





# In[7]:


labels[:100]


# In[ ]:





# In[8]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:





# In[1]:


import youtube_dl
ur = "https://www.youtube.com/watch?v=--swPW3U9EE"
ydl_opts = {
    'format': 'bestvideo[ext=mp4][height<=144]'
}
my_youtube_dl = youtube_dl.YoutubeDL(ydl_opts)
info_dict = my_youtube_dl.extract_info(ur, download=True)


# In[11]:


js


# In[ ]:





# In[ ]:





# In[8]:


import imageio
import cv2
import os
import shutil

def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
        

def get_frames(title,mypath,num):
    
    #print(title)
    
    vidcap = cv2.VideoCapture(title)
    
    success,image = vidcap.read()
    
    print(success)
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        try:
            cv2.imwrite( mypath + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        except Exception:
            pass
        count = count + 1
        if(count == num):
            success = False
    

            
get_frames("Texas Caviar (bean salad) Recipe - ADC Video---swPW3U9EE.mp4","K400/test/",10)


# In[ ]:




