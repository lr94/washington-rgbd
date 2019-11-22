import os
from shutil import copyfile
import time
from torch.hub import tqdm
from torch.nn import Module
import torch


def stopwatch(function):
    """
    This decorator allows to easily measure the execution time of a function:

    @stopwatch
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


def get_device(enable_cuda=True, cuda_device_id=None):
    """
    Get the device instance (GPU if available and enabled, otherwise CPU)

    :param enable_cuda:         Enable CUDA
    :param cuda_device_id:      Optionally specify which device to use
    :return:                    The device instance
    """
    if not enable_cuda and cuda_device_id is not None:
        raise ValueError("Invalid arguments")

    device = torch.device('cpu')
    device_name = "CPU"

    if enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda' + (':{}'.format(cuda_device_id) if cuda_device_id is not None else ''))
        device_name = torch.cuda.get_device_name(device)
    elif enable_cuda:
        print("CUDA not available, falling back on CPU")

    print("Device: {}".format(device_name))
    if device.index is not None:
        print("Device ID: {}".format(device.index))

    return device


class Logger:
    def __init__(self, period=1, use_tensorboard=False, tensorboard_logdir=None, model: Module = None,
                 model_save_path=None, log_file=None, input_shape=(1, 3, 224, 224)):
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
            # Tensorboard causes a lot of "FutureWarning", let's disable them
            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=tensorboard_logdir, flush_secs=5)

            dummy_input = torch.rand(input_shape).unsqueeze(0)
            self.writer.add_graph(model, dummy_input)
            self.writer.flush()

        self.step_counter = 0
        self.test_counter = 0
        self.log_file = log_file

        self.best_accuracy = 0

        self.counter = 0
        self._t = time.time()
        self.pb = None

    def __call__(self, epoch, batch_index, samples, batches, loss, batch_size, accuracy):
        """
        Update training logs
        :param epoch:                   Current epoch
        :param batch_index:             Index of the current batch during this epoch
        :param samples:                 Total number of samples in the training set
        :param batches:                 Total number of batches
        :param loss:                    Current value of loss function
        :param batch_size:              Current batch size
        """

        if self.pb is None:
            self.pb = tqdm(total=samples, unit=" samples")
        self.pb.update(batch_size)

        self.counter += 1
        if self.counter % self.period != 0:
            return

        t2 = time.time()

        if torch.is_tensor(loss):
            loss = loss.item()

        speed = (self.period * batch_size) / (t2 - self._t)
        self._print("Epoch {} ({:6} / {:6}, {:7.3f} %) Train loss: {:9.6f} Train accuracy: {:7.3f} Speed: {:8.3f} "
                    "samples/s".format(epoch, batch_index * batch_size, samples, batch_index / batches * 100.,
                                       loss, accuracy * 100., speed)
                    , step=self.step_counter
                    )

        if self.tensorboard:
            self.writer.add_scalar('train/loss', loss, self.step_counter)
            self.writer.add_scalar('train/speed', speed, self.step_counter)
            self.writer.add_scalar('train/accuracy', accuracy * 100., self.step_counter)
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
            self.pb.close()
            # self.pb.__exit__(None, None, None)
            self.pb = None

        accuracy = correct / total * 100.
        if self.tensorboard:
            self.writer.add_scalar('test/accuracy', accuracy, self.test_counter)
            self.writer.add_scalar('test/loss', loss, self.test_counter)
        self.test_counter += 1
        self._print("Accuracy: {:7.3f} % ({:6} / {:6}) Test loss: {:9.6f}".format(accuracy, correct, total, loss),
                    stdout=True)

    def start_epoch(self, epoch, epochs):
        """
        Logs the beginning of a new epoch
        :param epoch:       Epoch number
        :param epochs:      Total number of epochs
        """

        self._print("Epoch {} / {}".format(epoch, epochs), stdout=True)
        self._t = time.time()
        self.counter = 0
        if self.pb is not None:
            self.pb.close()
        self.pb = None

    def end_epoch(self, epoch, accuracy, loss):
        """
        To be called after the end of each epoch: saves the model and removes the old one (only after the new one has
        been saved)
        :param epoch:
        """

        if self.model is not None and self.model_save_path is not None:
            self._print("Saving model...", stdout=True)
            # Rename old model if present, in order to prevent data loss
            old = None
            if os.path.exists(self.model_save_path) and os.path.isfile(self.model_save_path):
                old = self.model_save_path + ".old"
                os.rename(self.model_save_path, old)

            data = {
                'm_state_dict': self.model.state_dict(),
                'epoch': epoch,

                'test_accuracy': accuracy,
                'test_loss': loss
            }
            torch.save(data, self.model_save_path)
            self._print("Model saved", stdout=True)
            # New model successfully saved, remove the old one if any
            if old is not None:
                os.remove(old)

            if accuracy > self.best_accuracy:
                self._print("Best accuracy achieved!", stdout=True)
                # TODO make it selectable
                best_path = '.'.join(self.model_save_path.split('.')[:-1]) + '_BEST.pth'
                copyfile(self.model_save_path, best_path)

                self.best_accuracy = accuracy

    def _print(self, *args, stdout=False, step=None):
        if stdout:
            print(*args)

        log_line = ''.join(map(str, args))

        if self.tensorboard:
            # TODO: this does not seem to work well to store and display textual logs, replace with something else
            # self.writer.add_text('log', log_line, global_step=step)
            pass

        if self.log_file is not None:
            with open(self.log_file, 'a+') as log_file_h:
                log_file_h.write(log_line + "\n")
