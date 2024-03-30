from parameters import *
from methods import *
from classes import *
import tqdm
from model import create_model
from tqdm.auto import tqdm
from datasets import (create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import MultiStepLR
import torch
import matplotlib.pyplot as plt
import time
import os
from torch.utils.tensorboard import SummaryWriter
plt.style.use('ggplot')
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# train
def train(train_data_loader, cur_model):

    print('Training')
    cur_model.train()

    # init tqdm
    progress_bar = tqdm(train_data_loader, total=len(train_data_loader))
    total_loss = 0
    for i, data in enumerate(progress_bar):
        try:
            optimizer.zero_grad()
            images, targets = data
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_list = cur_model(images, targets)
            losses = sum(loss for loss in loss_list.values())
            loss_value = losses.item()
            train_loss_hist.send(loss_value)
            losses.backward()
            optimizer.step()
            # update loss with progress bar
            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
            total_loss += loss_value  # Update total loss
        except Exception as e:
            print(f"Skipping batch due to error: {e}")

    return total_loss / len(train_data_loader)


# valid
def validate(valid_data_loader, cur_model):
    print('Validating')
    cur_model.eval()
    # init tqdm
    target = []
    predictions = []
    progress_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    for i, data in enumerate(progress_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            outputs = cur_model(images, targets)
        # mean Average Precision
        for i in range(len(images)):
            true_dict = dict()
            predictions_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            predictions_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            predictions_dict['scores'] = outputs[i]['scores'].detach().cpu()
            predictions_dict['labels'] = outputs[i]['labels'].detach().cpu()
            predictions.append(predictions_dict)
            target.append(true_dict)

    metric = MeanAveragePrecision()
    metric.update(predictions, target)
    metrics = metric.compute()
    return metrics


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_dataset = create_train_dataset(TRAIN_PATH, -1)
    valid_dataset = create_valid_dataset(VALID_PATH, -1)
    train_loader = create_train_loader(train_dataset, WORKERS)
    valid_loader = create_valid_loader(valid_dataset, WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    print("use " + device )
    # init the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES, size=RESIZE_TO)

    hparams = {
        'batch_size': BATCH_SIZE,
        'num_epochs': EPOCHS,
        'learning_rate': 0.0005,
        'momentum': 0.9,
        'resize_to': RESIZE_TO,
        'num_classes': NUM_CLASSES,
    }
    model = model.to(DEVICE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.0005, momentum=0.9, nesterov=True
    )
    scheduler = MultiStepLR(
        optimizer=optimizer, milestones=[45], gamma=0.1, verbose=True
    )
    # train loss
    train_loss_hist = LossTracker()
    # train loss + mean Average Precision
    train_loss_list = []
    map_50_list = []
    map_list = []
    # save the trained current model
    MODEL_NAME = 'model'
    # if show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from methods import plot_transformed_image
        plot_transformed_image(train_loader)
    # save best model until now
    save_best_model = BestModel()
    path_experiment = next_possible_dir('runs/' + EXPERIMENT)
    writer = SummaryWriter(path_experiment)
    # train loop
    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {EPOCHS}")
        train_loss_hist.reset()
        start = time.time()
        train_loss = train(train_loader, model)
        metric_summary = validate(valid_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} mAP@0.50:0.95: {metric_summary['map']}")
        print(f"Epoch #{epoch + 1} mAP@0.50: {metric_summary['map_50']}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        final_metrics = {
            'final_loss': train_loss,  # Or another metric
            'final_mAP': metric_summary['map'],
            'final_mAP_50': metric_summary['map_50'],
        }
        writer.add_scalar('final_loss', train_loss, epoch)
        writer.add_scalar('final_mAP',  metric_summary['map'], epoch)
        writer.add_scalar('final_mAP_50', metric_summary['map_50'], epoch)
        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        # save the best model untill now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # save current epoch model.
        save_model(epoch, model, optimizer)
        # loss
        save_loss_plot(OUT_PATH, train_loss_list)
        # mean Average Precision
        save_MAP(OUT_PATH, map_50_list, map_list)
        scheduler.step()

    writer.close()

