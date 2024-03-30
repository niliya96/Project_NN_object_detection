import numpy as np
import cv2
import glob as glob
import os
import time
import argparse
from model import create_model
from parameters import *

np.random.seed(42)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input',
    help='path to input image directory',
)
parser.add_argument(
    '--imgsz',
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
arguments = vars(parser.parse_args())

# dir for output images
os.makedirs('inference_outputs/images', exist_ok=True)

colors = [[0, 0, 0], [255, 0, 0]]

# best model  weights (from train)
best_model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.to(DEVICE).eval()

# dir for all images
TEST_PATH = arguments['input']
images_of_test = glob.glob(f"{TEST_PATH}/*.jpg")
print(f"Test instances: {len(images_of_test)}")

# total frames and total per seconds
total_frames = 0
total_fps = 0

for i in range(len(images_of_test)):
    # image file name for saving the output
    image_name = images_of_test[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(images_of_test[i])
    source_image = image.copy()
    if arguments['imgsz'] is not None:
        image = cv2.resize(image, (arguments['imgsz'], arguments['imgsz']))
    print(image.shape)
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # pixel range [0,1]]
    image /= 255.0
    # change color channels (H, W, C) => (C, H, W).
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # tensor
    image_input = torch.tensor(image_input, dtype=torch.float).cuda()
    image_input = torch.unsqueeze(image_input, 0)
    start_time = time.time()
    # predictions
    with torch.no_grad():
        outputs = best_model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Total FPS till current frame.
    total_fps += fps
    total_frames += 1

    # detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # for detected boxes only
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter boxes with threshold
        boxes = boxes[scores >= arguments['threshold']].astype(np.int32)
        draw_boxes = boxes.copy()
        # predicts classes names
        prediction_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # draw bbox and write classifications
        for j, box in enumerate(draw_boxes):
            class_name = prediction_classes[j]
            color = colors[CLASSES.index(class_name)]
            xmin = int((box[0] / image.shape[1]) * source_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * source_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * source_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * source_image.shape[0])
            cv2.rectangle(source_image,
                          (xmin, ymin),
                          (xmax, ymax),
                          color[::-1],
                          3)
            cv2.putText(source_image,
                        class_name,
                        (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color[::-1],
                        2,
                        lineType=cv2.LINE_AA)

        cv2.imshow('Prediction', source_image)
        cv2.waitKey(1)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", source_image)
    print(f"Image {i + 1} done...")
    print('-' * 50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# average FPS.
avg_fps = total_fps / total_frames
print(f"Average FPS: {avg_fps:.3f}")
