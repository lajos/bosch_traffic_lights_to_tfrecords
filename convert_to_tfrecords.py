#!/usr/bin/env python
#
# see configuration section in main() function
#
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import yaml
import os
from PIL import Image
import numpy as np
from math import floor, ceil
from io import BytesIO
from object_detection.utils import dataset_util
import tensorflow as tf
from tqdm import trange
import glob
from lxml import etree
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

labels = {}
n_labels = {}
MIN_BOX_WIDTH = 14

def clamp(value):
  return max(min(value, 1), 0)

def create_boxed_tf_example(img, box_data):
    global labels

    width,height = img.size
    image_format = b'jpg'

    filename = ''       # image data goes into tfrecord, no external file

    try:
        buffer = BytesIO()
        img.save(buffer,'jpeg')
        encoded_image_data = buffer.getvalue()
    except:
        return None

    xmins = box_data['xmins']
    xmaxs = box_data['xmaxs']
    ymins = box_data['ymins']
    ymaxs = box_data['ymaxs']
    classes_text = box_data['classes_text']
    classes = box_data['classes']

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def add_label(label):
    global labels, n_labels
    if label not in labels.keys():
        labels[label]=len(labels.values())+1

    if label not in n_labels.keys():
        n_labels[label] = 0

    n_labels[label] += 1


def add_from_yaml(yaml_path, writer, split_from=0., split_to=1., label_names=None ,ignore_occluded=True):
    global labels, n_labels

    print('adding images from yaml:', yaml_path)

    with open(yaml_path,'r') as yaml_stream:
        yaml_dirname = os.path.dirname(os.path.realpath(yaml_stream.name))
        yaml_data = yaml.load(yaml_stream)

    n_yaml_data = len(yaml_data)
    from_index = int(split_from * n_yaml_data)
    to_index = int(split_to * n_yaml_data)

    for i in trange(from_index, to_index):
        yaml_record = yaml_data[i]

        img_path = os.path.join(yaml_dirname, yaml_record['path']).replace('.png','.jpg')
        img = Image.open(img_path)
        width, height = img.size

        box_data = {'xmins':[],
                    'xmaxs':[],
                    'ymins':[],
                    'ymaxs':[],
                    'classes':[],
                    'classes_text':[]}

        for box in yaml_record['boxes']:
            # print(box)
            if ignore_occluded and box['occluded']:
                continue
            if label_names is None or box['label'] in label_names:
                if box['x_max']-box['x_min'] < 15:
                    continue
                box_data['xmins'].append(clamp(float(box['x_min'])/width))
                box_data['xmaxs'].append(clamp(float(box['x_max'])/width))
                box_data['ymins'].append(clamp(float(box['y_min'])/height))
                box_data['ymaxs'].append(clamp(float(box['y_max'])/height))

                label = box['label']

                add_label(label)

                box_data['classes_text'].append(label.encode('utf8'))
                box_data['classes'].append(labels[label])

            if not len(box_data['classes']):
                continue

            tf_example = create_boxed_tf_example(img, box_data)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())


def add_from_xml_dir(xml_dir, writer, label_names=None, split_from=0., split_to=1.):
    global labels

    print('adding images from xml dir:',xml_dir)

    xml_files = glob.glob(os.path.join(xml_dir,'*.xml'))

    n_xml_files = len(xml_files)
    from_index = int(split_from * n_xml_files)
    to_index = int(split_to * n_xml_files)

    for i in trange(from_index,to_index):
        xml_path = xml_files[i]
        xml_dirname = os.path.dirname(xml_path)

        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        img_name = os.path.basename(data['path'].replace('\\','/'))
        img_path = os.path.join(xml_dirname,img_name)
        # print(img_name,img_path)
        img = Image.open(img_path)

        width, height = img.size

        box_data = {'xmins':[],
                    'xmaxs':[],
                    'ymins':[],
                    'ymaxs':[],
                    'classes':[],
                    'classes_text':[]}

        for o in data['object']:
            box = o['bndbox']
            label = o['name']
            if label_names is None or label in label_names:
                if float(box['xmax'])-float(box['xmin']) <= MIN_BOX_WIDTH:
                    continue
                box_data['xmins'].append(clamp(float(box['xmin'])/width))
                box_data['xmaxs'].append(clamp(float(box['xmax'])/width))
                box_data['ymins'].append(clamp(float(box['ymin'])/height))
                box_data['ymaxs'].append(clamp(float(box['ymax'])/height))

                add_label(label)

                box_data['classes_text'].append(label.encode('utf8'))
                box_data['classes'].append(labels[label])

            if not len(box_data['classes']):
                continue

        tf_example = create_boxed_tf_example(img, box_data)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())


def write_label_map(label_map_path):
    global labels

    labels_rev = {}
    for k,v in labels.items():
        labels_rev[v]=k
    with open(label_map_path,'w') as lm_file:
        for label, label_name in labels_rev.items():
            lm_file.write('item {\n')
            lm_file.write('  id: {}\n'.format(label))
            lm_file.write("  name: '{}'\n".format(label_name))
            lm_file.write('}\n\n')


def main():
    # ---------------------------------------------------------------------------
    #
    # conversion settings
    #
    tfrecords_train_path = 'train.record'       # train tfrecords output path
    tfrecords_eval_path = 'eval.record'         # eval tfrecords output path
    label_map_path = 'label_map.pbtxt'          # label map output path
    train_split = 0.94                          # split for train (rest is eval)
    min_box_width = 14                          # boxes less wide are ignored
    label_names = ['Red', 'Yellow', 'Green']    # only add boxes with these labels

    # directory paths for PascalVOC xml files (for example output by https://github.com/tzutalin/labelImg)
    # xml files should be in the same folder as images
    #
    # list of directory path strings, example:
    # xml_dirs = ['../camera_images_labeled/1', '../traffic_light_images', '../udacity_succesful_light_detection']
    #
    xml_dirs=[]

    # path to bosch tl data set yaml file (https://hci.iwr.uni-heidelberg.de/node/6132)
    #
    # list of path strings, example:
    # yaml_paths = ['../dataset_train/train.yaml']
    #
    yaml_paths = []
    #
    # ---------------------------------------------------------------------------

    global n_labels
    global MIN_BOX_WIDTH
    MIN_BOX_WIDTH = min_box_width

    print('---- train records ----')
    split_from = 0
    split_to = train_split
    writer = tf.python_io.TFRecordWriter(tfrecords_train_path)
    n_labels={}
    for xml_dir in xml_dirs:
        add_from_xml_dir(xml_dir, writer, label_names=label_names, split_from=split_from, split_to=split_to)
    for yaml_path in yaml_paths:
        add_from_yaml(yaml_path, writer, label_names=label_names, split_from=split_from, split_to=split_to, ignore_occluded=True)
    writer.close()
    print('train n_labels:',n_labels)
    print('')


    print('---- eval records ----')
    split_from = train_split
    split_to = 1
    writer = tf.python_io.TFRecordWriter(tfrecords_eval_path)
    n_labels={}
    for xml_dir in xml_dirs:
        add_from_xml_dir(xml_dir, writer, label_names=label_names, split_from=split_from, split_to=split_to)
    for yaml_path in yaml_paths:
        add_from_yaml(yaml_path, writer, label_names=label_names, split_from=split_from, split_to=split_to, ignore_occluded=True)
    writer.close()
    print('eval n_labels:',n_labels)
    print('')


    global labels
    print('labels:',labels)
    write_label_map(label_map_path)



if __name__=='__main__':
    main()
