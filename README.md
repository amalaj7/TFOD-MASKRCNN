Tensorflow- MaskRCNN Steps
----------------------------------------

```bash
git clone https://github.com/amalaj7/TFOD-MASKRCNN.git
```

```bash
1.  conda create -n tfod python=3.6   
```
```bash
2.  conda activate tfod  
```
```bash
3.  pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.15.0 (for GPU- tensorflow-gpu)
```
```bash
4.  conda install -c anaconda protobuf   
```
```bash
5.  go to project path 'models/research'
```
```bash
6.  protoc object_detection/protos/*.proto --python_out=.  
```
```bash
7.  python setup.py install
```

### Install COCO API
```bash
8) pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### Resize images in a folder
```bash
9) python resize_images.py -d train_images/ -s 800 600
```

### Put images and annotations in corresponding folders inside images/ (Annotations are in COCO format)
```bash
10)  python create_coco_tf_record.py --logtostderr --train_image_dir=images/train_images --test_image_dir=images/test_images --train_annotations_file=coco_annotations/train.json --test_annotations_file=coco_annotations/test.json --include_masks=True --output_dir=./
```

* copy nets and deployment folder and export_inference_graph.py from slim folder and paste it in research folder 

Training
-----------------------------------------------------
* Create a folder called "training" , inside training folder download your custom model from [Model Zoo TF1](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) | [Model Zoo TF2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) , extract it and create a labelmap.pbtxt file(sample file is given in training folder) that contains the class labels
* Alterations in the config file , copy the config file from object_detection/samples/config and paste it in training folder or else u can use the pipeline.config that comes while downloading the pretrained model 
* Edit line no 10 - Number of classes
* Edit line no 128 - Path to model.ckpt file (downloaded model's file)
* Edit line no 134 - Iteration
* Edit line no 143 - path-to-train.record
* Edit line no 145 and 161 - path-to-labelmap
* Edit line no 159 - path to test.record

### Train model 
```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/mask_rcnn_resnet50_atrous_coco.config
```

### Export Tensorflow Graph
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/mask_rcnn_resnet50_atrous_coco.config --trained_checkpoint_prefix training/model.ckpt-10000 --output_directory my_model_mask
```

## Inference 
- Open object_detection_tutorial.ipynb and replace the necessary fields like model path, config path and test image path 

### Result
![Segmented Result](models/research/result2.png?raw=true "Title")

### View tensorboard
```bash
tensorboard --logdir=training
```

Tensorflow2 - MASKRCNN Steps
------------------------------------------

You can also follow this article: https://amalaj7.medium.com/maskrcnn-tensorflow-object-detection-api-8caae74ea4cc

- Almost similar steps as above .

```bash
git clone https://github.com/tensorflow/models.git
```

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### To test the installation
```bash
python object_detection/builders/model_builder_tf2_test.py
```

- Then follow the above steps from 8 to 10 (includes downloading the pretrained model and editing the config file according to your needs)

### Train the model
```bash
python model_main_tf2.py --pipeline_config_path=training/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config --model_dir=training --alsologtostderr

```

### View tensorboard
```bash
tensorboard --logdir=training
```

- Copy exporter_main_v2.py from object detection folder to research folder.

### Export Tensorflow Graph
```bash
python exporter_main_v2.py \
    --trained_checkpoint_dir training/model_checkpoint \
    --output_directory final_model \
    --pipeline_config_path training/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config
```

## Inference 
- For TFOD2 , you can utilize inference_from_saved_model_tf2_colab.ipynb and replace the necessary fields like model path, config path and test image path 

## TFLite Conversion
```bash
import tensorflow as tf

# Your saved model directory that contains graph
saved_model_dir = 'final_model/saved_model/'
```

```bash
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
# Save the tflite model in your research directory 
open("model.tflite", "wb").write(tflite_model)
```
