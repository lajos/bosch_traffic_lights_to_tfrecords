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

labels = {}

def create_tf_example(img, label):
    global labels
    height,width,channels = img.shape
    # print('image shape:',width, height)
    image_format = b'png'

    filename = ''

    try:
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer,'png')
        encoded_image_data = buffer.getvalue()
    except:
        return None

    xmins = [0]                # List of normalized left x coordinates in bounding box
    xmaxs = [1]                # List of normalized right x coordinates in bounding box
    ymins = [0]                # List of normalized top y coordinates in bounding box
    ymaxs = [1]                # List of normalized bottom y coordinates in bounding box
    classes_text = [label]     # List of string class name of bounding box
    classes = [labels[label]] # List of integer class id of bounding box

    # print('labels',labels)

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

def create_boxed_tf_example(img, img_path, box_data):
    global labels
    # width,height = img.size
    # print('image shape:',width, height)
    image_format = b'jpg'

    filename = img_path

    # buffer = BytesIO()
    # img.save(buffer,'jpeg')
    # encoded_image_data = buffer.getvalue()

    # try:
    #     buffer = BytesIO()
    #     img.save(buffer,'jpeg')
    #     encoded_image_data = buffer.getvalue()
    # except:
    #     return None

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_data = BytesIO(encoded_image)

    img2 = Image.open(encoded_image_data)
    width,height = img2.size
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

def get_image_boxes(yaml_record, yaml_path, ignore_occluded=False, label_names=None):
    global labels
    img_path = os.path.join(yaml_path, yaml_record['path'])
    img = Image.open(img_path)
    img_array = np.asarray(img)
    # print(img_array.shape)
    boxes = []
    for box in yaml_record['boxes']:
        # print(box)
        if ignore_occluded and box['occluded']:
            continue
        if label_names is None or box['label'] in label_names:
            img_cropped = img.crop((box['x_min'],
                                    box['y_max'],
                                    box['x_max'],
                                    box['y_min']))
            img_array_cropped = img_array[int(floor(box['y_min'])):int(ceil(box['y_max'])),
                                          int(floor(box['x_min'])):int(ceil(box['x_max']))]
            label = box['label']
            if label not in labels.keys():
                labels[label]=len(labels.values())+1
            boxes.append({'label':label, 'image':img_array_cropped})
    # print('labels:',labels)
    return boxes


def create_tf_record(yaml_data, yaml_path, tfrecords_path, label_map_path, split_from=0., split_to=1.):
    global labels

    n_yaml_data = len(yaml_data)
    from_index = int(split_from * len(yaml_data))
    to_index = int(split_to * len(yaml_data))

    writer = tf.python_io.TFRecordWriter(tfrecords_path)

    for i in trange(from_index, to_index):
        boxes = get_image_boxes(yaml_data[i], yaml_path, label_names=['Red', 'Green'])

        for i, box in enumerate(boxes):
            label = box['label']
            img = box['image']
            # print('label:',label)
            tf_example = create_tf_example(img, label)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    #print(boxes)

    # for i, box in enumerate(boxes):
    #     label = box['label']
    #     img = box['image']
    #     #box_array = np.asarray(img)
    #     print(img.shape)
    #     pil_img = Image.fromarray(img)
    #     pil_img.save('{}_{}.png'.format(i,label))
    #     buffer = BytesIO()
    #     pil_img.save(buffer,'png')
    #     img_data = buffer.getvalue()
    #     with open('test.png','wb') as s:
    #         s.write(img_data)


    writer.close()

    write_label_map(label_map_path)

    # print(labels)

def clamp(value):
  return max(min(value, 1), 0)


def create_boxed_tf_record(yaml_data, yaml_path, tfrecords_path, label_map_path, split_from=0., split_to=1., ignore_occluded=True):
    global labels

    n_yaml_data = len(yaml_data)
    from_index = int(split_from * len(yaml_data))
    to_index = int(split_to * len(yaml_data))

    writer = tf.python_io.TFRecordWriter(tfrecords_path)

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

            tf_example = create_boxed_tf_example(img, img_path, box_data)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                count += 1

    writer.close()
    print('count:',count)

    write_label_map(label_map_path)

    print(labels)


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

    # for i, yaml_record in enumerate(yaml_data):
    #     print(i, yaml_record)
    #print(yaml_data[9])
    # print(yaml_path)

    # create_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_train.tfrecords', label_map_path, split_from=0., split_to=0.8)
    # create_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_eval.tfrecords', label_map_path, split_from=0.8, split_to=1.)

    create_boxed_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_train.record', label_map_path, split_from=0., split_to=0.9)
    create_boxed_tf_record(yaml_data, yaml_path, tfrecords_prefix_path+'_eval.record', label_map_path, split_from=0.9, split_to=1.)

if __name__=='__main__':
    main2('train.yaml', 'bosch', 'bosch_label_map.pbtxt')
