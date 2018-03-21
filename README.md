## Convert the Bosch Traffic Lights Dataset to TFRecords

This script converts the [Bosch Small Traffic Lights Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) to [TFRecords](https://www.tensorflow.org/programmers_guide/datasets). It can also convert directories of Pascal VOC labeled images. Multiple datasets are merged into a single [TFRecords](https://www.tensorflow.org/programmers_guide/datasets) file.

Used this script to prepare data for training a [Tensorflow object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) model for the [Udacity](https://www.udacity.com/) self driving car nanodegree [System Integration Project](https://github.com/udacity/CarND-Capstone).

### Requirements

Using python2. I know. But [ROS](http://www.ros.org/) only supports python2, so it's what I had.

* [tensorflow](https://www.tensorflow.org/)
* [Pillow](https://pillow.readthedocs.io/en/latest/)
* [tqdm](https://github.com/noamraph/tqdm)
* [tensorflow models](https://github.com/tensorflow/models/tree/master)

### Bosch Dataset

Get the [Bosch Small Traffic Lights](https://hci.iwr.uni-heidelberg.de/node/6132) training dataset from [here](https://hci.iwr.uni-heidelberg.de/node/6132) and extract it. Add the `yaml` file path to the `yaml_paths` list in the `main()` function, for example:

```
yaml_paths = ['../dataset_train/train.yaml']
```

### Pascal VOC Labeled Images

To add Pascal VOC labeled images, save the `xml` files in the same directory as the source images. (A great tool to label images is [LabelImg](https://github.com/tzutalin/labelImg).) Add the directory to the `xml_dirs` list in in the `main()` function (can add multiple directories). For example:

```
xml_dirs = ['../camera_images_labeled/1', '../traffic_light_images', '../udacity_succesful_light_detection']
```

### Other Settings

Change the following settings in `main()` as needed:

```
tfrecords_train_path = 'train.record'       # train tfrecords output path
tfrecords_eval_path = 'eval.record'         # eval tfrecords output path
label_map_path = 'label_map.pbtxt'          # label map output path
train_split = 0.94                          # split for train (rest is eval)
min_box_width = 14                          # boxes less wide are ignored
label_names = ['Red', 'Yellow', 'Green']    # only add boxes with these labels
```

### Run

Make sure that the Tensorflow object detection directories are added to `PYTHONPATH` according to [this document](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath).

From terminal:

```
python2 convert_to_tfrecords.py
```

