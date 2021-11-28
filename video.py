from __future__ import division

from torch.autograd import Variable

from utils.utils import *
import argparse
import cv2
from models import load_model
import pickle as pkl

import random


def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="config/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="weights/yolov3.weights", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to     run detection on", default="video.mp4",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = 1
confidence = 0.5
nms_thesh = 0.5
start = 0
CUDA = torch.cuda.is_available()
num_classes = 80
classes = load_classes("coco.names")
model = load_model(args.cfgfile, args.weightsfile)
inp_dim = model.hyperparams['height']


if CUDA:
    model.cuda()

model.eval()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img





cap = cv2.VideoCapture(args.videofile)

frames = 0
start = time.time()


def prep_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = (inp_dim, inp_dim)
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim, inp_dim, 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    img = canvas
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (600, 600))
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img, volatile=True))
        output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

        if type(output) == int:
            frames += 1
            print("FPS {:5.4f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        classes = load_classes('coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame), output))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS {:5.2f}".format(frames / (time.time() - start)))
    else:
        break
