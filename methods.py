import os
import cv2
import albumentations as A
import numpy as np
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from parameters import DEVICE, CLASSES

plt.style.use('ggplot')

'''
Useful methods
'''


# Define the validation transforms.
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Handle data load and varying size tensors
def handle_data_load(batch):
    return tuple(zip(*batch))


# Handle training transforms.
def handle_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Blur(blur_limit=3, p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ToGray(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.3),
        A.RandomGamma(p=0.3),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


# Check if Bounded box is valid (x_right < x_max and y_min < y_right)
def is_bbox_valid(bbox):
    x_left, y_left, x_right, y_right, _ = bbox
    return x_left < x_right and y_left < y_right


# Return only valid boxes
def is_valid_bboxes(bboxes):
    return [bbox for bbox in bboxes if is_bbox_valid(bbox)]


# Check for the next possible dir and create a new dir
def next_possible_dir(base_dir):
    new_dir = base_dir
    dir_num = 1
    while os.path.exists(new_dir):
        dir_num += 1
        new_dir = base_dir + str(dir_num)
    return new_dir


# Shows the transformed images from the `train_loader`
# Depends on VISUALIZE_TRANSFORMED_IMAGES from parameters.py
def plot_transformed_image(train_loader):
    loop_times = 1
    if len(train_loader) > 0:
        for i in range(loop_times):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            # show
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]],
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Function to save the trained model
def save_model(epoch, model, optimizer):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'outputs/last_model.pth')


# Save both train loss graph
def save_loss_plot(OUT_DIR, train_loss_list, x_label='iterations', y_label='train loss', save_name='train_loss'):
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{OUT_DIR}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')


# Save Mean Average Precisions  0.5 and 0.95
def save_MAP(OUT_DIR, map_05, general_map):
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-',
        label='mAP@0.5'
    )
    ax.plot(
        general_map, color='tab:red', linestyle='-',
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/map.png")
