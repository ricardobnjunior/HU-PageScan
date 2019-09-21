import time
start = time.time()

import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from skimage.io import imsave
import skimage as sk
from skimage.io import imread
from skimage.measure import compare_ssim
from sklearn.metrics import roc_curve
import imutils
from sklearn.metrics import jaccard_similarity_score
from skimage.measure import _structural_similarity as ssim
import matplotlib.pylab as plt
import scikitplot as skplt
import sys
import os
import glob
import shutil
import re

backgrounds = ['background01', 'background02', 'background03', 'background04', 'background05']
documents = ['datasheet', 'letter', 'magazine', 'paper', 'patent', 'tax']
for bg in backgrounds:
    for doc in documents:

        unet_model = "checkpoint/pre_trained_model.pb"
        localization_save_output = "./output_"+bg+"/"+doc+"/"
        localization_save_output_file = "./output_"+bg+"/"+doc+"/result"+"/"
        type_data = doc
        images_path = 'D:/Ricardo/Datasets/validation/frames/'+bg+'/'+doc+'/'
        graph_path = "./"

        if not os.path.exists(localization_save_output):
            os.makedirs(localization_save_output)
        if not os.path.exists(localization_save_output_file):
            os.makedirs(localization_save_output_file)

        path_good = glob.glob(images_path)

        PATH_TO_FROZEN_GRAPH = graph_path+unet_model
        count1=0
        SAIDA=np.empty((1,512,512,1))
        ENTRADA=np.empty((1,512,512,1))
        GTIMG=np.empty((1,512,512,1))

        with tf.Session() as sess:
           print("load graph")
           with gfile.FastGFile(PATH_TO_FROZEN_GRAPH,'rb') as f:
               graph_def = tf.GraphDef()
           graph_def.ParseFromString(f.read())
           sess.graph.as_default()
           tf.import_graph_def(graph_def, name='')
           graph_nodes=[n for n in graph_def.node]
           names = []
           for t in graph_nodes:
              names.append(t.name)
           #print(names)
           # Get handles to # and output tensors
           ops = tf.get_default_graph().get_operations()
           all_tensor_names = {output.name for op in ops for output in op.outputs}

           tensor_dict = {}
           for key in [
               'conv2d_19/Sigmoid' ]:
               tensor_name = key + ':0'
               if tensor_name in all_tensor_names:
                   tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                      tensor_name)

           image_tensor = tf.get_default_graph().get_tensor_by_name('input_1:0')

           file_jaccard  = open(localization_save_output_file+'jaccard_'+type_data+'.txt', 'w')
           file_time  = open(localization_save_output_file+'time_run_'+type_data+'.txt', 'w')
           jaccard_result = []
           for path_caminho in path_good:
                for file in os.listdir(path_caminho):
                    if '_in' in file:
                        impath = images_path+file

                        teste = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)

                        altura_img_original = teste.shape[0]
                        largura_img_original = teste.shape[1]
                        teste = 1 - (teste/255)
                        teste = cv2.resize(teste, (512, 512), interpolation=cv2.INTER_CUBIC)
                        teste = (teste.reshape((512, 512, 1)))

                        ENTRADA[0,:,:,0] = cv2.resize(teste,(512,512))
                        start_antes_run = time.time()


                        output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: ENTRADA})
                        end_depois_run = time.time()

                        time_pross = end_depois_run - start_antes_run
                        file_time.write(str(time_pross)+'\n')

                        SAIDA=output_dict['conv2d_19/Sigmoid']

                        saida = sk.img_as_ubyte(SAIDA[0,:]) #convert for 8 bits

                        saida = cv2.resize(saida, (largura_img_original, altura_img_original), interpolation=cv2.INTER_CUBIC)

                        result_name = file.replace('_in','_result')
                        cv2.imwrite(localization_save_output+result_name, saida)

                        y_true_path = impath.replace('_in','_gt')
                        y_true = cv2.imread(y_true_path)
                        y_pred = cv2.imread(localization_save_output+result_name)

                        jaccardScore = jaccard_similarity_score(y_true.flatten(), y_pred.flatten())
                        file_jaccard.write(str(jaccardScore)+'\n')



        file_jaccard.close()
        file_time.close()


