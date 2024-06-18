import os, sys, time, shutil
import pandas as pd
import numpy as np
import imageio.v3 as iio
import cv2
import h5py
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Choose and create data set

# inputs:
#   filename is csv file with the Matt Groh Fitzpatrick 17k images
#   c is an integer for the minimum number of images in a pair-wise group of disease label and partition
def choose_diseases(filename, per_disease_cutoff, num_train, num_test):
  df = pd.read_csv(filename)
  # group into 3 partitions (1 & 2, 3 & 4, 5 & 6)
  df['olina_scale'] = df['fitzpatrick_centaur'].apply( lambda x: 1 if ((x > 0) & (x < 3)) else 2 if x < 5 else 3 )
  df['combo_label'] = df['label'] + '_' + df['olina_scale'].astype(str)
  # find labels with enough images per skin tone
  unique_labels = df['label'].unique()
  labels_to_use = []
  for lab in unique_labels:
    locations = df.loc[df['label'] == lab]
    if len(locations) > per_disease_cutoff:
      lengths = [len(locations.loc[locations['olina_scale'] == x]) for x in range(1, 4)]
      if min(lengths) > num_train + num_test:
        labels_to_use.append(lab)
        print(lab, lengths)
        lengths = [len(locations.loc[locations['fitzpatrick_centaur'] == x]) for x in range(1, 7)]
        print(lengths)
  print(labels_to_use, len(labels_to_use))

  ###### NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE 
  print('limiting labels_to_use')
  labels_to_use = ('folliculitis', 'squamous cell carcinoma', 'lichen planus')
  print(labels_to_use, len(labels_to_use))
  df_to_use = df.loc[df['label'].isin(labels_to_use)].copy()
  df_to_use.reset_index(inplace=True)
  print(df_to_use.groupby(['combo_label']).size())
  #df_to_use.to_csv('x.csv')
  #df_to_use['file'] = df['url'].str.split('/').str[-1]
  unique_labels = list(df_to_use.label.unique()) # note this is going to be a rather small number
  unique_combo_labels = list(df.combo_label.unique())
  df_to_use['c'] = ''
  df_to_use['y'] = ''
  df_to_use['type'] = 'have enough'
  y_col_idx = df_to_use.columns.get_loc('y')
  c_col_idx = df_to_use.columns.get_loc('c')
  type_col_idx = df_to_use.columns.get_loc('type')
  for disease in unique_labels:
    for tone in (1, 3): # ignore skin tone olina scale of 2
      this_combo_label = disease + '_' + str(tone)
      num_need_imgs = per_disease_cutoff + num_test - len(df_to_use.loc[df_to_use.combo_label == this_combo_label])
      if num_need_imgs > 0:
        i = 0
        for index, row in df_to_use.loc[df_to_use.combo_label == this_combo_label].iterrows():
          df_to_use.iloc[index, y_col_idx] = unique_labels.index(row.label)
          df_to_use.iloc[index, c_col_idx] = unique_combo_labels.index(row.combo_label)
          if i < num_test:
            df_to_use.iloc[index, type_col_idx] = 'test'
          elif i < num_test + 5:
            df_to_use.iloc[index, type_col_idx] = 'dreambooth'
          else:
            df_to_use.iloc[index, type_col_idx] = 'train'
          i += 1
  df_to_use.to_csv('df.csv')
  return df_to_use

def save_h5py(x_tr, y_tr, c_tr, o_tr, x_te, y_te, c_te, o_te, file):
  path = os.path.dirname(file)
  if not os.path.exists(path):
    os.mkdir(path)
  db = h5py.File(file, 'w')
  db.create_dataset('x_tr', x_tr.shape, dtype=np.float32, data=x_tr)
  db.create_dataset('x_te', x_te.shape, dtype=np.float32, data=x_te)
  db.create_dataset('y_tr', y_tr.shape, dtype=np.uint8, data=y_tr)
  db.create_dataset('y_te', y_te.shape, dtype=np.uint8, data=y_te)
  db.create_dataset('c_tr', c_tr.shape, dtype=np.uint8, data=c_tr)
  db.create_dataset('c_te', c_te.shape, dtype=np.uint8, data=c_te)
  db.create_dataset('o_tr', o_tr.shape, dtype=np.uint8, data=o_tr)
  db.create_dataset('o_te', o_te.shape, dtype=np.uint8, data=o_te)
  db.close()
  print('saved', file)

# inputs:
#  df
#  folder
#  resize
#  crop
#   num_test is the number of images used for testing, rest are used to train
#   num_train is not needed as the rest are used for training
def load_dataset(df, folder, resize, crop, num_test):
  labels = list(df.label.unique())
  print(labels)
  combo_labels = list(df.combo_label.unique())
  x = np.zeros((len(df), resize, resize, 3), dtype=np.uint8)
  y = np.zeros(len(df), dtype=np.uint8)
  c = np.zeros(len(df), dtype=np.uint8)
  o = np.zeros(len(df), dtype=np.uint8)
  for index, row in df.iterrows():
    img_name = os.path.join(folder, df.loc[df.index[index], 'md5hash'] + '.jpg')
    img = iio.imread(img_name)
    img = cv2.resize(img, (resize, resize))
    x[index] = img
    y[index] = labels.index(row.label)
    c[index] = combo_labels.index(row.combo_label)
    o[index] = row.olina_scale
  x_tr, y_tr, c_tr, o_tr = [], [], [], []
  x_te, y_te, c_te, o_te = [], [], [], []
  for cl in combo_labels:
    mask = df['combo_label'] == cl
    x_te.append(np.asarray(x[mask][0:num_test]))
    x_tr.append(np.asarray(x[mask][num_test:]))
    y_te.append(np.asarray(y[mask][0:num_test]))
    y_tr.append(np.asarray(y[mask][num_test:]))
    c_te.append(np.asarray(c[mask][0:num_test]))
    c_tr.append(np.asarray(c[mask][num_test:]))
    o_te.append(np.asarray(o[mask][0:num_test]))
    o_tr.append(np.asarray(o[mask][num_test:]))
  x_tr = np.concatenate(x_tr, axis=0)
  y_tr = np.concatenate(y_tr, axis=0)
  c_tr = np.concatenate(c_tr, axis=0)
  o_tr = np.concatenate(o_tr, axis=0)
  x_te = np.concatenate(x_te, axis=0)
  y_te = np.concatenate(y_te, axis=0)
  c_te = np.concatenate(c_te, axis=0)
  o_te = np.concatenate(o_te, axis=0)
  return x_tr, y_tr, c_tr, o_tr, x_te, y_te, c_te, o_te

# inputs:
#   rootdir is the location in which there is a fitzpatrick17k and an images subfolder
#   num_test is the number of images used for testing, rest are used to train
#   num_train is not needed as the rest are used for training
def load_and_create_h5py(df, mg_root_dir, num_set, num_train, num_test):
  start = time.time()
  x_tr, y_tr, c_tr, o_tr, x_te, y_te, c_te, o_te = load_dataset(df, os.path.join(mg_root_dir, 'data/finalfitz17k'), resize, crop, num_test)
  end = time.time()
  print('load_dataset', round(end-start, 2))
  start = end
  save_h5py(x_tr, y_tr, c_tr, o_tr, x_te, y_te, c_te, o_te, h5pyFile)
  end = time.time()
  print('save_h5py', round(end-start, 2))
  start = end

# df should be the the subset with the diseases that were picked by choose_diseases
def set_up_dreambooth_training(df, rootdir, num_desired_imgs, num_train, num_test):
  unique_labels = df['label'].unique() # note this is going to be a rather small number
  unique_combo_labels = df['combo_label'].unique()
  train_dir = os.path.join(rootdir, 'sd_training_images')
  if not os.path.exists(train_dir):
    os.mkdir(train_dir)
  os.chdir(train_dir)
  for disease in unique_labels:
    if not os.path.exists(disease):
      os.mkdir(disease)
    os.chdir(disease)
    for tone in range(1, 4):
      if not os.path.exists(str(tone)):
        os.mkdir(str(tone))
      os.chdir(str(tone))
      disease_tone = df.loc[(df['label'] == disease) & (df['olina_scale'] == tone)]
      this_combo_label = disease + '_' + str(tone)
      check = df.loc[df['combo_label'] == this_combo_label]
      train_disease_tone = disease_tone.loc[disease_tone.type == 'dreambooth']
      assert(len(disease_tone) == len(check))
      num_need_imgs = num_desired_imgs + num_test - len(disease_tone)
      images = []
      if num_need_imgs > 0:
        print(disease, tone, num_need_imgs)
        for index, row in train_disease_tone.iterrows():
          print(row.label, row.olina_scale, row.type, row.url)
          print(os.getcwd(), os.path.exists('../../../images/' + os.path.basename(row.url)))
          images.append(Image.open('../../../images/' + os.path.basename(row.url)))
          if len(images) == 5:
            break
        [image.save(f"{i}.jpeg") for i, image in enumerate(images)]
      # return back to disease folder
      os.chdir('..')
    # return back to training images folder
    os.chdir('..')

if __name__ == '__main__':
  num_train = 5
  num_test = 35
  per_disease_cutoff = 200 # diseases with total images fewer than this are ignored
  resize = 256
  crop = 224
  start = time.time()
  h5pyFile = '/scratch/olina/cutoff40.h5'
  rootdir = './omu2324'
  mg_rootdir = os.path.join(rootdir, 'mattgroh')
  df = choose_diseases(os.path.join(mg_rootdir, 'fitzpatrick17k.csv'), per_disease_cutoff, num_train, num_test)
  end = time.time()
  print('choose_diseases took', round(end-start, 2))
  start = end
  set_up_dreambooth_training(df, rootdir, per_disease_cutoff, num_train, num_test)
  end = time.time()
  print('load_and_create_dreambooth took', round(end-start, 2))
  start = end
  load_and_create_h5py(df, mg_rootdir, per_disease_cutoff, num_train, num_test)
  end = time.time()
  print('load_and_create_h5py took', round(end-start, 2))
  start = end
