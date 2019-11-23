import time
import os
from shutil import copyfile

from argparse import ArgumentParser

from torch.hub import tqdm
from torch.nn import Module, DataParallel
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


class Logger:
    def __init__(self, use_tensorboard=False, tensorboard_logdir=None, model: Module = None,
                 model_save_path=None, log_file=None, input_shape=(1, 3, 224, 224)):
        """
        :param use_tensorboard:     Whether to enable Tensorboard logging
        :param model:               Model
        :param model_save_path:     Path to save model after each epoch
        """

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
            if model is not None and not isinstance(model, DataParallel):
                self.writer.add_graph(model, dummy_input)
            self.writer.flush()

        self.step_counter = 0
        self.test_counter = 0
        self.log_file = log_file

        self.best_accuracy = 0

        self.counter = 0
        self.processed_samples = 0
        self.last_output_processed_samples = 0
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

        # Base 1
        batch_index += 1

        # Init progress bar if not already initialized
        if self.pb is None:
            self.pb = tqdm(total=samples, unit=" samples", leave=False)
        # Update progress bar
        self.pb.update(batch_size)

        # Update batch counter and sample counter
        self.counter += 1
        self.processed_samples += batch_size

        t2 = time.time()

        if torch.is_tensor(loss):
            loss = loss.item()

        if batch_index == batches:
            speed = (self.processed_samples - self.last_output_processed_samples) / (t2 - self._t)
            self.last_output_processed_samples = self.processed_samples
            self._print("Epoch {} ({:6} / {:6}, {:7.3f} %) Train loss: {:9.6f} Train accuracy: {:7.3f} Speed: {:8.3f} "
                        "samples/s".format(epoch, self.processed_samples, samples, batch_index / batches * 100.,
                                           loss, accuracy * 100., speed), step=self.step_counter)

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
            self.pb = tqdm(total=total, unit=" samples", leave=False)

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
        self.processed_samples = 0
        self.last_output_processed_samples = 0
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

            m = self.model.module if isinstance(self.model, DataParallel) else self.model
            data = {
                'm_state_dict': m.state_dict(),
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


def get_device(enable_cuda=True, cuda_device_ids=None):
    if not enable_cuda and cuda_device_ids is not None:
        raise ValueError("Invalid arguments")

    # We will return the first device in the list
    first_id = cuda_device_ids
    if first_id is not None and isinstance(first_id, list):
        first_id = first_id[0]

    # Prepare CPU fallback
    device = torch.device('cpu')
    device_name = "CPU"

    # If CUDA is enabled and available
    if enable_cuda and torch.cuda.is_available():
        # Get device and its name
        device = torch.device('cuda' + (':{}'.format(first_id) if first_id is not None else ''))
        device_name = torch.cuda.get_device_name(device)
    elif enable_cuda:
        # CPU fallback
        enable_cuda = False
        print("CUDA not available, falling back on CPU")

    # If only one device print its name
    if cuda_device_ids is None or not enable_cuda:
        print("Device: {}".format(device_name))
    else:
        # If multiple devices print all their names
        for i in cuda_device_ids:
            device_name = torch.cuda.get_device_name(torch.device('cuda:{}'.format(i)))
            print("Device {}: {}".format(i, device_name))

    return device


def init_device_model(args, model: torch.nn.Module = None, model_file=None):
    device = get_device(enable_cuda=not args.disable_cuda, cuda_device_ids=args.cuda_device)
    batch_size = args.batch_size

    last_epoch = 0
    if model is not None:
        if model_file is not None and os.path.exists(model_file):
            print("Loaded")
            data = torch.load(model_file, map_location=device)
            model.load_state_dict(data['m_state_dict'])
            last_epoch = data['epoch']

        if torch.cuda.device_count() > 1:
            model = DataParallel(model, args.cuda_device)

        model = model.to(device)

    return device, model, batch_size, last_epoch


def add_device_options(parser: ArgumentParser):
    device_opt_g = parser.add_argument_group(title="Device options")
    device_opt_g.add_argument('--disable-cuda', action='store_true', help="Disable GPU acceleration")
    device_opt_g.add_argument('--cuda-device', nargs='+', type=int, help="Select a specific GPU")
    device_opt_g.add_argument('--batch-size', type=int, default=64, help="Batch size for training and testing")
