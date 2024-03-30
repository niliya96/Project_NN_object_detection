import torch

'''
Useful classes
'''


# Saves the current best model. Will save the model state if the current epoch's
#       validation mAP @0.5:0.95 IoU higher than the previous max
class BestModel:
    def __init__(
            self, best_valid_map=float(0)
    ):
        self.best_valid_map = best_valid_map

    def __call__(
            self,
            model,
            current_valid_map,
            epoch,
            OUT_DIR,
    ):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_maps
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, f"{OUT_DIR}/best_model.pth")


# Keeps track of the train and valid loss values
# Saves average for each epoch
class LossTracker:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
