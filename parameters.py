import torch

'''
Paths and dirs
'''
# train data
TRAIN_PATH = 'data/train'
# valid data
VALID_PATH = 'data/valid'
# save model and other outputs
OUT_PATH = 'outputs'

'''
Parameters about the classes of the data
'''
CLASSES = ['shared-taxi', 'truck-5-axle', 'auto', 'mav', 'private-bus', 'taxi', 'cycle', 'truck-4-axle', 'mini-lcv', 'lcv',
           'goods-auto', 'heavy-vehicle', 'govt-bus', 'BIKE', 'truck-3-axle', 'school-bus', 'tractor', 'two-wheeler', 'mini-bus', 'pedestrian', 'car', 'truck-2-axle']
NUM_CLASSES = len(CLASSES)

'''
Hyper parameters
'''
# epochs
EPOCHS = 70
# batches
BATCH_SIZE = 32
# resize the images to this size for transform
RESIZE_TO = 640
# parallel workers that would load the data
WORKERS = 2
# TODO
IMAGES_TRAIN = -1

'''
Other parameters
'''
# use to create the path to the experiment files
EXPERIMENT = "regular_mobilenets"
# CPU/GPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# visualize images after create the data loaders (init to false)
VISUALIZE_TRANSFORMED_IMAGES = False



