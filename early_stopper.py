
class EarlyStopper:
    """
    Early stopping to stop the training if the validation loss stops improving.
    Arguments:
        min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        patience: int, number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
