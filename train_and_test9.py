import os, sys, time, shutil
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import math
import h5py
import sklearn
import imageio.v3 as iio
import cv2
import argparse

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

initial_learning_rate = 0.000001
def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)

def setup_resnet18(img_size, num_classes):
  """
  Reference:
  [1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
  [2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
  Surpassing human-level performance on imagenet classification. In
  ICCV, 2015.
  """

  class ResnetBlock(tf.keras.Model):
    #A standard resnet block.
    def __init__(self, channels: int, down_sample=False):
      # channels: same as number of convolution kernels
      super().__init__()
      
      self.__channels = channels
      self.__down_sample = down_sample
      self.__strides = [2, 1] if down_sample else [1, 1]
      
      KERNEL_SIZE = (3, 3)
      # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
      INIT_SCHEME = "he_normal"

      self.conv_1 = tf.keras.layers.Conv2D(self.__channels, strides=self.__strides[0],
                           kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
      self.bn_1 = tf.keras.layers.BatchNormalization()
      self.conv_2 = tf.keras.layers.Conv2D(self.__channels, strides=self.__strides[1],
                           kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
      self.bn_2 = tf.keras.layers.BatchNormalization()
      self.merge = tf.keras.layers.Add()

      if self.__down_sample:
        # perform down sampling using stride of 2, according to [1].
        self.res_conv = tf.keras.layers.Conv2D(
          self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
        self.res_bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
      res = inputs
      
      x = self.conv_1(inputs)
      x = self.bn_1(x)
      x = tf.nn.relu(x)
      x = self.conv_2(x)
      x = self.bn_2(x)

      if self.__down_sample:
        res = self.res_conv(res)
        res = self.res_bn(res)

      # if not perform down sample, then add a shortcut directly
      x = self.merge([x, res])
      out = tf.nn.relu(x)
      return out


  class ResNet18(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
      # num_classes: number of classes in specific classification task.
      super().__init__(**kwargs)
      self.conv_1 = tf.keras.layers.Conv2D(64, (7, 7), strides=2,
                           padding="same", kernel_initializer="he_normal")
      self.init_bn = tf.keras.layers.BatchNormalization()
      self.pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
      self.res_1_1 = ResnetBlock(64)
      self.res_1_2 = ResnetBlock(64)
      self.res_2_1 = ResnetBlock(128, down_sample=True)
      self.res_2_2 = ResnetBlock(128)
      self.res_3_1 = ResnetBlock(256, down_sample=True)
      self.res_3_2 = ResnetBlock(256)
      self.res_4_1 = ResnetBlock(512, down_sample=True)
      self.res_4_2 = ResnetBlock(512)
      self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
      self.flat = tf.keras.layers.Flatten()
      self.fc = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
      out = self.conv_1(inputs)
      out = self.init_bn(out)
      out = tf.nn.relu(out)
      out = self.pool_2(out)
      for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
        out = res_block(out)
      out = self.avg_pool(out)
      out = self.flat(out)
      out = self.fc(out)
      return out

  model = ResNet18(num_classes)
  model.build(input_shape=(None, img_size[0], img_size[1], img_size[2]))
  model.compile(optimizer="adam",
                loss='binary_crossentropy', metrics=["accuracy"])
  return model
  #model.summary()

def setup_model(len_labels, xsize, ysize, x, y):
  y_cat = tf.keras.utils.to_categorical(y, len(np.unique(y)))
  baseModel = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_tensor=tf.keras.Input(shape=(xsize, ysize, 3)))
  #baseModel = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=tf.keras.Input(shape=(xsize, ysize, 3)))
  #baseModel.summary()
  baseModel.trainable = False
  #baseModel = setup_resnet18((xsize, ysize, 3), len(np.unique(y)))
  model = tf.keras.models.Sequential()
  model.add(baseModel)
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(4096, activation='relu')) # from VGG paper
  model.add(tf.keras.layers.Dense(4096, activation='relu')) # from VGG paper
  model.add(tf.keras.layers.Dense(256, activation=None)) # from MG paper
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.Dense(len(np.unique(y)), activation='softmax'))
  model.build(input_shape = (None, xsize, ysize, 3))
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  #model.summary()

  # lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(decay_steps=0.01,decay_rate=)
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
  #print('auc', m.result().numpy())
  return(m.result().numpy(), tp.result().numpy(), tn.result().numpy(), fp.result().numpy(), fn.result().numpy())
  #result = confusion_matrix(y_test, y_prediction , normalize='pred')
  #result = sklearn.metrics.confusion_matrix(y_cat, y_pred, normalize='pred')
  #print(result)

def read_dreambooth_output_by_disease(root_dir, disease, disease_value, resize, debug):
  cwd = os.getcwd()
  list_of_x = []
  list_of_y = []
  for tone in (1, 3): # limit tones to 1 and 3
    subdir = disease + '_' + str(tone)
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
        x[index]= img
        y[index] = disease_value
      list_of_x.append(x)
      list_of_y.append(y)
    else:
      if debug > 10:
        print('not found', subdir)
  x = np.concatenate(list_of_x)
  y = np.concatenate(list_of_y)
  os.chdir(cwd)
  return x, y
  return None, None

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
      x[index]= img
      y[index] = disease_value
  else:
    x, y = [], []
    if debug > 10:
      print('not found', subdir)
  os.chdir(cwd)
  return x, y


def zip_shuffle(x, y):
  z = list(zip(x, y))
  np.random.shuffle(z)
  shuffle_x, shuffle_y = zip(*z)
  return np.asarray(shuffle_x), np.asarray(shuffle_y)

def train_and_test(x_tr, y_tr, x_te, y_te):
  print(len(x_tr), len(y_tr))
  '''
  print('y', np.unique(y_tr), np.unique(y_te))
  print('c', np.unique(c_tr), np.unique(c_te))
  print('o', np.unique(o_tr), np.unique(o_te))
  print('initial_learning_rate', initial_learning_rate)
  '''
  start=time.time()
  num_categories = len(np.unique(y_tr))
  print('num of categories', num_categories)

  model = setup_model(len(np.unique(y_tr)), resize, resize, x_tr, y_tr)
  end = time.time()
  print('train time', round(end-start, 2))
  start = end
  a, tp, tn, fp, fn = test(model, x_te, y_te, num_categories)
  print('testing all size', len(y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  end = time.time()
  print('test all time', round(end-start, 2))
  start = end
  for i in np.unique(y_te):
    mask = np.isin(y_te, i)
    my_x_te = x_te[mask]
    my_y_te = y_te[mask]
    a, tp, tn, fp, fn = test(model, my_x_te, my_y_te, num_categories)
    print('testing disease of', i, 'size', len(my_y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  for i in (1, 3): # limit tones to 1 and 3
    mask = np.isin(o_te, i)
    my_x_te = x_te[mask]
    my_y_te = y_te[mask]
    a, tp, tn, fp, fn = test(model, my_x_te, my_y_te, num_categories)
    print('testing olina_scale of', i, 'size', len(my_y_te), ': auc', a, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
  end = time.time()
  print('test subset time', round(end-start, 2))
  start = end



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='fitzpatrick')
  parser.add_argument('--num', type = int, required = False, default = 0)
  parser.add_argument('--debug', type = int, required = False, default = 0)
  args = parser.parse_args()
  print(args)
  
  if os.path.exists('/content'):
    if not os.path.exists('/content/drive'):
      from google.colab import drive
      drive.mount('/content/drive')
    if not os.path.exists('/content/fitzpatrick17k.csv'):
      shutil.copy('/content/drive/MyDrive/Science Fair 2023-2024/fitzpatrick17k.csv', '/content/fitzpatrick17k.csv')
    if not os.path.exists('/content/cutoff40.h5'):
      shutil.copy('/content/drive/MyDrive/Science Fair 2023-2024/cutoff40.h5', '/content/cutoff40.h5')
    clean = False
    rootdir = '/content'
  else:
    rootdir = '/scratch/content'

  disease_dict = {'folliculitis': 0, 'squamous': 1, 'lichen': 2}

  resize = 256
  start = time.time()
  orig_x_tr, orig_y_tr, c_tr, orig_o_tr, x_te, y_te, c_te, o_te = read_h5py(os.path.join(rootdir, 'cutoff40.h5'))
  end = time.time()
  print('load', round(end-start, 2))

  x_tr = orig_x_tr
  y_tr = orig_y_tr
  print(len(x_tr), len(y_tr), np.unique(y_tr), np.unique(c_tr), np.unique(orig_o_tr))
  dbout_root = '/afs/ece/usr/tamal/.vol2/omu2324/control/dreambooth_output'
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

  '''
  add_x_tr_f_1, add_y_tr_f_1 = read_new_disease_tone(dbout_root, 'folliculitis', 0, 1, 256, args.debug)
  add_x_tr_f_3, add_y_tr_f_3 = read_new_disease_tone(dbout_root, 'folliculitis', 0, 3, 256, args.debug)
  add_x_tr_s_1, add_y_tr_s_1 = read_new_disease_tone(dbout_root, 'squamous', 1, 1, 256, args.debug)
  add_x_tr_s_3, add_y_tr_s_3 = read_new_disease_tone(dbout_root, 'squamous', 1, 3, 256, args.debug)
  add_x_tr_l_1, add_y_tr_l_1 = read_new_disease_tone(dbout_root, 'lichen', 2, 1, 256, args.debug)
  add_x_tr_l_3, add_y_tr_l_3 = read_new_disease_tone(dbout_root, 'lichen', 2, 1, 256, args.debug)

  #  threshold - 200 - number images created = number of images to add to skin-tone-disease combo
  # if negative, don't add any

  # the number of images needed to get to 200 images for disease
  if ((args.num < len(add_y_tr_f)) & (args.num < len(add_y_tr_s)) & (args.num < len(add_y_tr_l))):
    x = np.concatenate((x_tr, add_x_tr_f[:args.num], add_x_tr_s[:args.num], add_x_tr_l[:args.num]))
    y = np.concatenate((y_tr, add_y_tr_f[:args.num], add_y_tr_s[:args.num], add_y_tr_l[:args.num]))
  else:
    print('all possible data')
    x = np.concatenate((x_tr, add_x_tr_f, add_x_tr_s, add_x_tr_l))
    y = np.concatenate((y_tr, add_y_tr_f, add_y_tr_s, add_y_tr_l))
  '''

  x, y = zip_shuffle(x_tr, y_tr)
  train_and_test(x, y, x_te, y_te)



  
