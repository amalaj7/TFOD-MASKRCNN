Tensorflow- MaskRCNN Steps
----------------------------------------
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
4.  conda install -c anaconda protobuf   # to convert protocolbuffer(language ofGoogle to py)
```
```bash
5.  go to project path 'research'
```
```bash
6.  protoc object_detection/protos/*.proto --python_out=.  (to convert proto to .py)
```
```bash
7.  python setup.py install
```

### Install COCO API
8) pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

### Resize images in a folder
9) python resize_images.py -d train_images/ -s 800 600

### Put images and annotations in corresponding folders inside images/ 
10)  python create_coco_tf_record.py --logtostderr --train_image_dir=images/train_images --test_image_dir=images/test_images --train_annotations_file=coco_annotations/train.json --test_annotations_file=coco_annotations/test.json --include_masks=True --output_dir=./

* copy nets and deployment folder and export_inference_graph.py from slim folder and paste it in research folder 

Training
-----------------------------------------------------
* Create a folder called "training" ,inside training folder download ur custom model from model zoo,extract it and create a labelmap.pbtxt file that contains the class labels
* Alterations in the config file , copy the config file from object_detection/samples/config and paste it in training folder or else u can use the pipeline.config that comes while downloading the pretrained model 
* Edit line no 10 (acc. to ur number of classes)
* Edit line no 128 (acc. to ur downloaded models ckpt file)
* Edit line no 134 - Iteration
* Edit line no 143 - path-to-train.record
* Edit line no 145 and 161 - path-to-labelmap
* Edit line no 159 - path to test.record

```bash
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/mask_rcnn_resnet50_atrous_coco.config
```
```bash
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/mask_rcnn_resnet50_atrous_coco.config --trained_checkpoint_prefix training/model.ckpt-200 --output_directory my_model_mask
```

### View tensorboard
```bash
tensorboard --logdir=training
```