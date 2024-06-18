import os
import sys
import time
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import math
import h5py
import imageio.v3 as iio
import cv2
import argparse

initial_learning_rate = 0.000001
def lr_exp_decay(epoch, lr):
  k = 0.1
  return initial_learning_rate * math.exp(-k*epoch)

def setup_model(len_labels, xsize, ysize, x, y):
  y_cat = tf.keras.utils.to_categorical(y, len(np.unique(y)))
  baseModel = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=tf.keras.Input(shape=(xsize, ysize, 3)))
  #baseModel.summary()
  baseModel.trainable = False
  model = tf.keras.models.Sequential()
  model.add(baseModel)
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(4096, activation='relu')) # from VGG paper
  model.add(tf.keras.layers.Dense(4096, activation='relu')) # from VGG paper
  model.add(tf.keras.layers.Dense(256, activation=None)) # from MG paper
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.Dense(len(np.unique(y)), activation='softmax'))
  model.build(input_shape = (None, xsize, ysize, 3))
  #model.summary()
  model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
  batch_size = 128
  epochs=32
  model.fit(x, y_cat, verbose=0, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=0)],)
  return model



def test(model, x, y, num_categories):
  y_cat = tf.keras.utils.to_categorical(y, num_categories) 
  m = tf.keras.metrics.AUC(num_thresholds=200, curve="ROC", summation_method="interpolation", name=None, dtype=None, thresholds=None, multi_label=False, num_labels=None, label_weights=None, from_logits=False)
  tp = tf.keras.metrics.TruePositives()
  tn = tf.keras.metrics.TrueNegatives()
  fp = tf.keras.metrics.FalsePositives()
  fn = tf.keras.metrics.FalseNegatives()
  
  m.reset_state()
  tp.reset_state()
  tn.reset_state()
  fp.reset_state()
  fn.reset_state()
  y_pred = np.argmax(model.predict(x, verbose=0), axis=1)
  m.update_state(y, y_pred)
  tp.update_state(y, y_pred)
  tn.update_state(y, y_pred)
  fp.update_state(y, y_pred)
  fn.update_state(y, y_pred)

  return(m.result().numpy(), tp.result().numpy(), tn.result().numpy(), fp.result().numpy(), fn.result().numpy())

def read_h5py(file):
  db = h5py.File(file, 'r')
  x_tr = np.array(db['x_tr'])
  x_te = np.array(db['x_te'])
  y_tr = np.array(db['y_tr'])
  y_te = np.array(db['y_te'])
  c_tr = np.array(db['c_tr'])
  c_te = np.array(db['c_te'])
  o_tr = np.array(db['o_tr'])
  o_te = np.array(db['o_te'])

  return x_tr, y_tr, c_tr, o_tr, x_te, y_te, c_te, o_te

def zip_shuffle(x, y):
  z = list(zip(x, y))
  np.random.shuffle(z)
  shuffle_x, shuffle_y = zip(*z)

  return np.asarray(shuffle_x), np.asarray(shuffle_y)


def read_new_disease_tone(root_dir, disease, disease_value, tone_value, resize, debug):
  cwd = os.getcwd()
  subdir = disease + '_' + str(tone_value)
  if os.path.exists(os.path.join(root_dir, subdir)):
    if debug > 10:
      print('found', subdir)
    os.chdir(os.path.join(root_dir, subdir))
    files = os.listdir()
    x = np.zeros((len(files), resize, resize, 3), dtype=np.uint8)
    y = np.zeros(len(files), dtype=np.uint8)
    for index, file in enumerate(files):
      img = iio.imread(file)
      img = cv2.resize(img, (resize, resize))
      x[index], y[index] = img, disease_value
  else:
    x, y = [], []
    if debug > 10:
      print('not found', subdir)
  os.chdir(cwd)
  return x, y

def train_and_test(x_tr, y_tr, x_te, y_te):
  print(len(x_tr), len(y_tr))
  start=time.time()
  num_categories = len(np.unique(y_tr))
  print('num of categories', num_categories)
  model = setup_model(len(np.unique(y_tr)), resize, resize, x_tr, y_tr)
  print('train time', round(time.time()-start, 2))
  start = time.time()
  a, tp, tn, fp, fn = test(model, x_te, y_te, num_categories)
  print('testing all size', len(y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  print('test all time', round(time.time()-start, 2))
  start = time.time()
  for i in np.unique(y_te):
    mask = np.isin(y_te, i)
    my_x_te, my_y_te = x_te[mask], y_te[mask]
    a, tp, tn, fp, fn = test(model, my_x_te, my_y_te, num_categories)
    print('testing disease of', i, 'size', len(my_y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  for i in (1, 3): # limit tones to 1 and 3
    mask = np.isin(o_te, i)
    my_x_te, my_y_te = x_te[mask], y_te[mask]
    a, tp, tn, fp, fn = test(model, my_x_te, my_y_te, num_categories)
    print('testing olina_scale of', i, 'size', len(my_y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  print('test subset time', round(time.time()-start, 2))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='fitzpatrick')
  parser.add_argument('--num', type = int, required = False, default = 0)
  parser.add_argument('--debug', type = int, required = False, default = 0)
  args = parser.parse_args()
  print(args)
  
  rootdir = '/scratch/content'
  disease_dict = {'folliculitis': 0, 'squamous': 1, 'lichen': 2}

  resize = 256
  start = time.time()
  orig_x_tr, orig_y_tr, c_tr, orig_o_tr, x_te, y_te, c_te, o_te = read_h5py(os.path.join(rootdir, 'cutoff40.h5'))
  print('load', round(time.time()-start, 2))

  x_tr, y_tr = orig_x_tr, orig_y_tr
  print(len(x_tr), len(y_tr), np.unique(y_tr), np.unique(c_tr), np.unique(orig_o_tr))
  dbout_root = './omu2324/control/dreambooth_output'
  for val, dis in enumerate(disease_dict):
    for tone_val in (1, 3):
      maskd = np.isin(orig_y_tr, val)
      maskt = np.isin(orig_o_tr, tone_val)
      mask = maskd & maskt
      add_x_tr, add_y_tr = read_new_disease_tone(dbout_root, dis, val, tone_val, 256, args.debug)
      need = args.num - mask.sum()
      if need < 0:
        add_x_tr = []
        add_y_tr = []
      else:
        if len(add_x_tr) > need:
          add_x_tr = add_x_tr[:need]
          add_y_tr = add_y_tr[:need]
        else:
          pass # use all of the additional images
      print(dis, val, tone_val, mask.sum(), need, len(add_y_tr))
      if (len(add_y_tr) > 0):
        x_tr = np.concatenate((x_tr, add_x_tr))
        y_tr = np.concatenate((y_tr, add_y_tr))

  x, y = zip_shuffle(x_tr, y_tr)
  train_and_test(x, y, x_te, y_te)

  
