import numpy as np
import cv2
import os
import time
import argparse
import pathlib
from model import create_model
from parameters import *

np.random.seed(42)

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='data/testing.mp4'
)
parser.add_argument(
    '--imgsz',
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.7,
    type=float,
    help='detection threshold'
)
args = vars(parser.parse_args())

os.makedirs('inference_outputs/videos', exist_ok=True)

colors = [[0, 0, 0], [255, 0, 0]]

# best model  weights (from train)
best_model = create_model(num_classes=NUM_CLASSES, size=640)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.to(DEVICE).eval()

# detection threshold.
detection_threshold = 0.2

cap = cv2.VideoCapture(args['input'])

if not cap.isOpened():
    print('Error while trying to read video. Please check path again')

# Get the frame width and height.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
print(save_name)
# Define codec and create VideoWriter object .
out = cv2.VideoWriter(f"inference_outputs/videos/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

# Read until end of video.
while cap.isOpened():
    # Capture each frame of the video.
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if args['imgsz'] is not None:
            image = cv2.resize(image, (args['imgsz'], args['imgsz']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # pixel range [0,1]]
        image /= 255.0
        # change color channels (H, W, C) => (C, H, W).
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # tensor.
        image_input = torch.tensor(image_input, dtype=torch.float).cuda()
        image_input = torch.unsqueeze(image_input, 0)
        start_time = time.time()
        # predictions
        with torch.no_grad():
            outputs = best_model(image_input.to(DEVICE))
        end_time = time.time()

        # current fps.
        fps = 1 / (end_time - start_time)
        # total FPS until current frame.
        total_fps += fps
        frame_count += 1

        # detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # for detected boxes only
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`.
            boxes = boxes[scores >= args['threshold']].astype(np.int32)
            draw_boxes = boxes.copy()
            # predictions class names.
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw bbox and write classifications
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                try:
                    color_index = CLASSES.index(class_name)
                    color = colors[color_index]
                # handle exception: class name is not in colors
                except IndexError:
                    # random color
                    color = np.random.randint(0, 255, size=3).tolist()
                # scale boxes
                xmin = int((box[0] / image.shape[1]) * frame.shape[1])
                ymin = int((box[1] / image.shape[0]) * frame.shape[0])
                xmax = int((box[2] / image.shape[1]) * frame.shape[1])
                ymax = int((box[3] / image.shape[0]) * frame.shape[0])
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color[::-1], 3)
                cv2.putText(frame, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[::-1], 2,
                            lineType=cv2.LINE_AA)
        out.write(frame)

    else:
        break

cap.release()

# average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
