import os
import time
from torch.hub import tqdm
from torch.nn import Module
from torch import save, is_tensor

# Tensorboard causes a lot of "FutureWarning", let's disable them
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter


def measure(function):
    """
    This decorator allows to easily measure the execution time of a function:

    @measure
    def function(argument):
        some_object.heavy_computation(argument)

    Now calling function(...) will also print on stdout the number of milliseconds elapsed
    """

    def function_wrapper(*args, **kwargs):
        start = time.time()
        ret = function(*args, **kwargs)
        end = time.time()
        print('{}() took {:.3f} ms'.format(function.__name__, 1000. * (end - start)))
        return ret
    return function_wrapper


class Logger:
    def __init__(self, period=1, use_tensorboard=False, model: Module = None, model_save_path=None):
        """
        :param period:              Number of batches between each update
        :param use_tensorboard:     Whether to enable Tensorboard logging
        :param model:               Model
        :param model_save_path:     Path to save model after each epoch
        """

        self.period = period
        self.model = model
        self.model_save_path = model_save_path
        self.tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(flush_secs=5)
        self.step_counter = 0
        self.test_counter = 0

        self.counter = 0
        self._t = time.time()
        self.pb = None

    def reset_epoch(self):
        """
        To be called at the beginning of each epoch to reset the bach counter
        """

        self._t = time.time()
        self.counter = 0

    def __call__(self, epoch, batch_index, samples, batches, loss, batch_size):
        """
        Update training logs
        :param epoch:                   Current epoch
        :param batch_index:             Index of the current batch during this epoch
        :param samples:                 Total number of samples in the training set
        :param batches:                 Total number of batches
        :param loss:                    Current value of loss function
        :param batch_size:              Current batch size
        """

        self.counter += 1
        if self.counter % self.period != 0:
            return

        t2 = time.time()

        if is_tensor(loss):
            loss = loss.item()

        speed = (self.period * batch_size) / (t2 - self._t)
        print("Epoch {} ({:6} / {:6}, {:6.3f} %) Train loss: {:9.6f} Speed: {:8.3f} samples/s".format(
            epoch, batch_index * batch_size, samples, batch_index / batches * 100.,
            loss, speed
        ))

        if self.tensorboard:
            self.writer.add_scalar('train/loss', loss, self.step_counter)
            self.writer.add_scalar('train/speed', speed, self.step_counter)
        self.step_counter += 1

        self._t = t2

    def log_test_progress(self, processed, total):
        """
        This method can be called during model evaluation to display progress
        :param processed:           Number of processed samples
        :param total:               Total number of samples in the test set
        """

        if self.pb is None:
            self.pb = tqdm(total=total, unit=" samples")

        self.pb.update(processed - self.pb.n)

    def log_test_result(self, correct, total, loss):
        """
        Log test loss and accuracy
        :param correct:             Number of correct predictions
        :param total:               Total number of samples in the test set
        :param loss:                Loss
        """

        if self.pb is not None:
            self.pb.__exit__(None, None, None)
            self.pb = None

        accuracy = correct / total * 100.
        if self.tensorboard:
            self.writer.add_scalar('test/accuracy', accuracy, self.test_counter)
            self.writer.add_scalar('test/loss', loss, self.test_counter)
        self.test_counter += 1
        print("Accuracy: {:6.3f} % ({:6} / {:6}) Test loss: {:9.6f}".format(accuracy, correct, total, loss))

    def start_epoch(self, epoch, epochs):
        """
        Logs the beginning of a new epoch
        :param epoch:       Epoch number
        :param epochs:      Total number of epochs
        """

        print("Epoch {} / {}".format(epoch, epochs))

    def end_epoch(self, epoch):
        """
        To be called after the end of each epoch: saves the model and removes the old one (only after the new one has
        been saved)
        :param epoch:
        """

        if self.model is not None and self.model_save_path is not None:
            print("Saving model...")
            old = None
            if os.path.exists(self.model_save_path) and os.path.isfile(self.model_save_path):
                old = self.model_save_path + ".old"
                os.rename(self.model_save_path, old)

            save(self.model.state_dict(), self.model_save_path)
            print("Model saved")
            if old is not None:
                os.remove(old)
