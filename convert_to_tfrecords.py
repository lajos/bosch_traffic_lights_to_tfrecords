#!/usr/bin/env python

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

labels = {}

def clamp(value):
  return max(min(value, 1), 0)

def create_boxed_tf_example(img, box_data):
    global labels

    width,height = img.size
    # print('image shape:',width, height)
    image_format = b'jpg'

    filename = ''       # image data goes into tfrecord, no external file

    # buffer = BytesIO()
    # img.save(buffer,'jpeg')
    # encoded_image_data = buffer.getvalue()

    try:
        buffer = BytesIO()
        img.save(buffer,'jpeg')
        encoded_image_data = buffer.getvalue()
    except:
        return None

    # with tf.gfile.GFile(img_path, 'rb') as fid:
    #     encoded_image = fid.read()
    # encoded_image_data = BytesIO(encoded_image)

    # img2 = Image.open(encoded_image_data)
    # width,height = img2.size
    # print('image shape:',width, height)

    xmins = box_data['xmins']
    xmaxs = box_data['xmaxs']
    ymins = box_data['ymins']
    ymaxs = box_data['ymaxs']
    classes_text = box_data['classes_text']
    classes = box_data['classes']

    # print(box_data)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_boxed_tf_record(yaml_data, yaml_path, tfrecords_path, label_map_path, split_from=0., split_to=1., ignore_occluded=True):
    global labels

    n_yaml_data = len(yaml_data)
    from_index = int(split_from * len(yaml_data))
    to_index = int(split_to * len(yaml_data))

    label_names=['Red', 'Green']

    count = 0

    for i in trange(from_index, to_index):
        yaml_record = yaml_data[i]

        img_path = os.path.join(yaml_path, yaml_record['path']).replace('.png','.jpg')
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
                if label not in labels.keys():
                    labels[label]=len(labels.values())+1

                box_data['classes_text'].append(label.encode('utf8'))
                box_data['classes'].append(labels[label])

            if not len(box_data['classes']):
                continue

            # print(img_path)
            # print(box_data)

            tf_example = create_boxed_tf_example(img, box_data)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                count += 1

    print('count:',count)

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


def main2(yaml_filename, tfrecords_prefix_path, label_map_path):
    with open(yaml_filename,'r') as yaml_stream:
        yaml_path = os.path.dirname(os.path.realpath(yaml_stream.name))
        yaml_data = yaml.load(yaml_stream)

    create_boxed_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_train.record', label_map_path, split_from=0., split_to=0.9)
    create_boxed_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_eval.record', label_map_path, split_from=0.9, split_to=1.)


def add_from_xml_dir(xml_dir, writer, split_from, split_to):
    xml_files = glob.glob(os.path.join(xml_dir,'*.xml'))

    for i in trange(len(xml_files)):
        xml_path = xml_files[i]
        xml_dirname = os.path.dirname(xml_path)

def main():
    tfrecords_train_path = 'train.record'       # train tfrecords output path
    tfrecords_eval_path = 'eval.record'         # eval tfrecords output path
    label_map_path = 'label_map.pbtxt'          # label map output path
    train_split = 0.8                           # normalized percentage to use for train (rest is eval)
    xml_dirs = ['../camera_images_labeled/1']

    # write train records
    split_from = 0
    split_to = train_split
    writer = tf.python_io.TFRecordWriter(tfrecords_train_path)
    for xml_dir in xml_dirs:
        add_from_xml_dir(xml_dir, writer, split_from, split_to)
    writer.close()

    global labels
    print('labels:',labels)
    write_label_map(label_map_path)



if __name__=='__main__':
    main()
