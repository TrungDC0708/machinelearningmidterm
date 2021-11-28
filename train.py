from __future__ import division

import os
import argparse

from models import load_model
from utils.logger import Logger
from utils.utils import *
from utils.datasets import DataModule

import numpy as np
from torch.autograd import Variable


def parse_data_config(path):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def run():
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="coco128/coco2.data", help="Path to data config file (.data)")
    parser.add_argument("--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    logger = Logger(args.logdir)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(args.data)
    model = load_model(args.model, args.pretrained_weights)

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    data_module = DataModule(data_config, mini_batch_size, model.hyperparams['height'])
    dataloader = data_module.train_dataloader()

    validation_dataloader = data_module.val_dataloader()

    optimizer = model.configure_optimizers()

    for epoch in range(20):
        print("\n---- Training Model ----")
        model.train()
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i
            train_batch = imgs, targets
            loss, loss_components = model.training_step(train_batch, batch_i)
            loss.backward()

            if batches_done % model.hyperparams['subdivisions'] == 0:
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                for g in optimizer.param_groups:
                    g['lr'] = lr
                optimizer.step()
                optimizer.zero_grad()

            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        if epoch % 1 == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.weights"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            model.save_weights(checkpoint_path)

        if epoch % 1 == 0:
            print("\n---- Evaluating Model ----")
            model.eval()
            labels = []
            sample_metrics = []
            for _, imgs, targets in tqdm.tqdm(validation_dataloader, desc="Validating"):
                labels += targets[:, 1].tolist()
                targets[:, 2:] = xywh2xyxy(targets[:, 2:])
                targets[:, 2:] *= model.hyperparams['height']
                imgs = Variable(imgs.type(torch.cuda.FloatTensor), requires_grad=False)
                with torch.no_grad():
                    outputs = model(imgs)
                    outputs = non_max_suppression(outputs, conf_thres=0.1, iou_thres=0.5)
                sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)

            true_positives, pred_scores, pred_labels = [
                np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            metrics_output = ap_per_class(
                true_positives, pred_scores, pred_labels, labels)
            precision, recall, AP, f1, ap_class = metrics_output
            print(f"---- mAP {AP.mean():.5f} ----")

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    run()
