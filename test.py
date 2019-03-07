#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
import random
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')

#tf.enable_eager_execution()


# In[2]:


train_df = pd.read_csv('data/train_labels.csv')


# In[3]:


#images_list = os.listdir('data/train/')
images_list = train_df['id'].values
labels = train_df['label'].values


# In[4]:


images_list = 'data/train/' + train_df['id'] + '.tif'
images_list = images_list.values

path_ds = tf.data.Dataset.from_tensor_slices(images_list)


# In[75]:


def preprocess_image(image):
    #image = tf.image.decode_jpeg(image, channels=3)
    #print('******************************', str(image.numpy()))
    import pdb; pdb.set_trace()
    #image = tf.cast(mpimg.imread(image), tf.float32)
    #image = tf.image.resize_images(image, [192, 192])
    #image /= 255.0  # normalize to [0,1] range
    return tf.constant('lol')
    #return image

def load_and_preprocess_image(path):
    pdb.set_trace()
    image = tf.read_file(path)
    return preprocess_image(image)


AUTOTUNE = 1 #tf.data.experimental.AUTOTUNE# = 1
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


# In[66]:


#a = tf.Tensor('lol', dtype=str)
#print(kk)


# In[15]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
for n,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))


# In[ ]:


label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


# In[72]:


mammal = tf.Tensor("Elephant", dtype=tf.string)
mammal


# In[73]:


get_ipython().run_line_magic('pinfo', 'tf.Tensor')


# In[13]:


test_df = pd.read_csv('data/sample_submission.csv')
get_ipython().run_line_magic('pinfo', 'tf.image.decode_image')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info(), test_df.info()


# In[ ]:


img = mpimg.imread('data/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif')


# In[ ]:


img.shape


# In[ ]:


plt.imshow(img)


# In[ ]:


# plot some random training images with their ground truth labels
i = random.choice(range(train_df.shape[0]))
img = mpimg.imread('data/train/' + train_df.iloc[i]['id'] + '.tif')
plt.imshow(img)
plt.title(train_df.iloc[i]['label'])
plt.show()


# In[ ]:


train_df['label'].value_counts()


# In[ ]:


train_df['label'].hist


# In[ ]:


img = mpimg.imread('data/train/dd6dfed324f9fcb6f93f46f32fc800f2ec196be2.tif')
plt.imshow(img)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




