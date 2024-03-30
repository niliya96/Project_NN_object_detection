import tqdm
from tqdm import tqdm
from parameters import *
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader


def validate(valid_data_loader, best_model):
    print('Validating')
    best_model.eval()

    # init tqdm progress bar.
    predictions = []
    target = []
    progress_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    for i, data in enumerate(progress_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = best_model(images, targets)

        # mean Average Precision
        for i in range(len(images)):
            true_dict = dict()
            predictions_list = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            predictions_list['boxes'] = outputs[i]['boxes'].detach().cpu()
            predictions_list['scores'] = outputs[i]['scores'].detach().cpu()
            predictions_list['labels'] = outputs[i]['labels'].detach().cpu()
            predictions.append(predictions_list)
            target.append(true_dict)

    metric = MeanAveragePrecision()
    metric.update(predictions, target)
    metrics_list = metric.compute()
    return metrics_list


if __name__ == '__main__':
    # load best model that trained and weights
    model = create_model(num_classes=NUM_CLASSES, size=640)
    output_model_path = 'outputs/best_model.pth'
    checkpoint = torch.load(output_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    test_images_dataset = 'data/Test/Test/JPEGImages'
    test_dataset = create_valid_dataset(test_images_dataset)
    test_loader = create_valid_loader(test_dataset, num_workers=WORKERS)

    metrics = validate(test_loader, model)
    print(f"mAP_50: {metrics['map_50'] * 100:.3f}")
    print(f"mAP_50_95: {metrics['map'] * 100:.3f}")
