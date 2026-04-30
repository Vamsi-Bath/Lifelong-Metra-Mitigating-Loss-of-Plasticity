# built-in libraries
import time
import os
import pickle
from copy import deepcopy
import json
import argparse
from functools import partialmethod

# third party libraries
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import (
    ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
)
from mlproj_manager.file_management.file_and_directory_management import store_object_with_several_attempts

from lop.nets.torchvision_modified_resnet import build_resnet18, kaiming_init_resnet_module
from lop.algos.res_gnt import ResGnT
from lop.incremental_cifar.invarianceCIFAR import compute_invariance_scores_cifar


def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices.
    """
    idx = sub_sample_indices.numpy()
    cifar_data.data["data"] = cifar_data.data["data"][idx]
    cifar_data.data["labels"] = cifar_data.data["labels"][idx]
    cifar_data.integer_labels = torch.tensor(cifar_data.integer_labels)[idx].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        debug = access_dict(exp_params, key="debug", default=False, val_type=bool)
        turn_off_debugging_processes(debug)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=self.verbose)

        random_seeds = get_random_seeds()
        self.random_seed = random_seeds[self.run_index]
        torch.random.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

        self.data_path = exp_params["data_path"]
        self.num_workers = access_dict(exp_params, key="num_workers", default=1, val_type=int)

        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]

        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network."))
        self.early_stopping = access_dict(exp_params, "early_stopping", default=False, val_type=bool)

        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert (not self.use_cbp) or (self.replacement_rate > 0.0), "Replacement rate should be greater than 0."
        self.utility_function = access_dict(
            exp_params, "utility_function", default="weight", val_type=str,
            choices=["weight", "contribution"]
        )
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert (not self.use_cbp) or (self.maturity_threshold > 0), "Maturity threshold should be greater than 0."

        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator = self.noise_std > 0.0

        self.num_epochs = access_dict(exp_params, "num_epochs", default=4000, val_type=int)
        self.num_classes = access_dict(exp_params, "num_classes", default=100, val_type=int)
        self.current_num_classes = access_dict(exp_params, "initial_num_classes", default=5, val_type=int)
        self.class_increment = access_dict(exp_params, "class_increment", default=5, val_type=int)
        self.class_increase_frequency = access_dict(
            exp_params, "class_increase_frequency", default=200, val_type=int
        )

        self.batch_sizes = {
            "train": access_dict(exp_params, "train_batch_size", default=128, val_type=int),
            "test": access_dict(exp_params, "test_batch_size", default=256, val_type=int),
            "validation": access_dict(exp_params, "validation_batch_size", default=256, val_type=int),
        }

        self.image_dims = (32, 32, 3)
        self.num_train_samples_per_class = access_dict(
            exp_params, "num_train_samples_per_class", default=450, val_type=int
        )
        self.num_val_samples_per_class = access_dict(
            exp_params, "num_val_samples_per_class", default=50, val_type=int
        )

        self.activation = access_dict(
            exp_params, key="activation", default="relu", val_type=str,
            choices=["relu", "tanh", "crelu"]
        )

        self.net = build_resnet18(
            num_classes=self.num_classes,
            norm_layer=torch.nn.BatchNorm2d,
            activation=self.activation,
        )
        self.net.apply(kaiming_init_resnet_module)

        self.optim = torch.optim.SGD(
            self.net.parameters(),
            lr=self.stepsize,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.net.to(self.device)
        self.current_epoch = 0

        self.resgnt = None
        if self.use_cbp:
            self.resgnt = ResGnT(
                net=self.net,
                hidden_activation=self.activation,
                replacement_rate=self.replacement_rate,
                decay_rate=0.99,
                util_type=self.utility_function,
                maturity_threshold=self.maturity_threshold,
                device=self.device,
            )

        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.best_accuracy_model_parameters = {}

        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency
        self.delete_old_checkpoints = True

        self.running_avg_window = access_dict(exp_params, "running_avg_window", default=1, val_type=int)
        self.current_running_avg_step = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0

        self.invariance_eval_batch_size = access_dict(
            exp_params, key="invariance_eval_batch_size", default=256, val_type=int
        )
        self.invariance_firing_rate = access_dict(
            exp_params, key="invariance_firing_rate", default=0.01, val_type=float
        )

        print("Loaded config:")
        print("  num_epochs =", self.num_epochs)
        print("  num_classes =", self.num_classes)
        print("  initial_num_classes =", self.current_num_classes)
        print("  class_increment =", self.class_increment)
        print("  class_increase_frequency =", self.class_increase_frequency)
        print("  batch_sizes =", self.batch_sizes)
        print("  num_workers =", self.num_workers)
        print("  train/val samples per class =", self.num_train_samples_per_class, self.num_val_samples_per_class)

        self._initialize_summaries()

    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment.
        Use maximum task size so arrays are large enough for the full run.
        """
        steps_per_epoch = int(
            np.ceil((self.num_classes * self.num_train_samples_per_class) / self.batch_sizes["train"])
        )
        checkpoints_per_epoch = int(np.ceil(steps_per_epoch / self.running_avg_window))
        total_checkpoints = max(1, self.num_epochs * checkpoints_per_epoch)

        train_prototype_array = torch.zeros(total_checkpoints, device=self.device, dtype=torch.float32)
        self.results_dict["train_loss_per_checkpoint"] = torch.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = torch.zeros_like(train_prototype_array)

        prototype_array = torch.zeros(self.num_epochs, device=self.device, dtype=torch.float32)
        self.results_dict["epoch_runtime"] = torch.zeros_like(prototype_array)

        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = torch.zeros_like(prototype_array)
            self.results_dict[set_type + "_invariance_per_epoch"] = torch.zeros_like(prototype_array)

        self.results_dict["class_order"] = self.all_classes

    def get_experiment_checkpoint(self):
        """
        Creates a dictionary with all the necessary information to pause and resume the experiment.
        """
        partial_results = {}
        for k, v in self.results_dict.items():
            partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()

        checkpoint = {
            "model_weights": self.net.state_dict(),
            "optim_state": self.optim.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        if self.use_cbp:
            checkpoint["resgnt"] = self.resgnt

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values.
        """
        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        self.net.load_state_dict(checkpoint["model_weights"])
        self.optim.load_state_dict(checkpoint["optim_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])

        self.current_epoch = checkpoint["epoch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        partial_results = checkpoint["partial_results"]
        for k in self.results_dict.keys():
            val = partial_results[k]
            self.results_dict[k] = val if not isinstance(val, torch.Tensor) else val.to(self.device)

        if self.use_cbp and "resgnt" in checkpoint:
            self.resgnt = checkpoint["resgnt"]

    def _store_training_summaries(self):
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += (
            self.running_loss / self.running_avg_window
        )
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += (
            self.running_accuracy / self.running_avg_window
        )

        self._print("\t\tOnline accuracy: {0:.4f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _compute_invariance_on_loader(self, data_loader: DataLoader) -> float:
        """
        Computes mean invariance score on a single batch from the provided loader.
        """
        try:
            sample = next(iter(data_loader))
        except StopIteration:
            return 0.0

        x_eval = sample["image"].to(self.device)
        x_eval = x_eval[:self.invariance_eval_batch_size]

        use_abs = (self.activation == "crelu")

        with torch.no_grad():
            layer_scores, _ = compute_invariance_scores_cifar(
                net=self.net,
                x_eval=x_eval,
                use_abs=use_abs,
                firing_rate=self.invariance_firing_rate,
            )

        if len(layer_scores) == 0:
            return 0.0

        return float(np.mean(layer_scores))

    def _store_test_summaries(self, test_data: DataLoader, val_data: DataLoader, epoch_number: int, epoch_runtime: float):
        """
        Computes test summaries and stores them in results dir.
        """
        self.results_dict["epoch_runtime"][epoch_number] += torch.tensor(epoch_runtime, dtype=torch.float32)

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self.evaluate_network(data_loader)
            invariance = self._compute_invariance_on_loader(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best and accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_accuracy_model_parameters = deepcopy(self.net.state_dict())

            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += torch.tensor(
                evaluation_time, dtype=torch.float32
            )
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += loss
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += accuracy
            self.results_dict[data_name + "_invariance_per_epoch"][epoch_number] += torch.tensor(
                invariance, dtype=torch.float32, device=self.device
            )

            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))
            self._print("\t\t{0} invariance: {1:.4f}".format(data_name, invariance))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data: DataLoader):
        """
        Evaluates the network on the test data.
        """
        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0

        with torch.no_grad():
            for _, sample in enumerate(test_data):
                images = sample["image"].to(self.device)
                test_labels = sample["label"].to(self.device)
                test_predictions = self.net.forward(images)[:, self.all_classes[:self.current_num_classes]]

                avg_loss += self.loss(test_predictions, test_labels)
                avg_acc += torch.mean(
                    (test_predictions.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32)
                )
                num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    def run(self):
        training_data, training_dataloader = self.get_data(train=True, validation=False)
        val_data, val_dataloader = self.get_data(train=True, validation=True)
        test_data, test_dataloader = self.get_data(train=False)

        self.load_experiment_checkpoint()

        self.train(
            train_dataloader=training_dataloader,
            test_dataloader=test_dataloader,
            val_dataloader=val_dataloader,
            test_data=test_data,
            training_data=training_data,
            val_data=val_data
        )

    def get_data(self, train: bool = True, validation: bool = False):
        """
        Loads the data set.
        """
        cifar_data = CifarDataSet(
            root_dir=self.data_path,
            train=train,
            cifar_type=100,
            device=None,
            image_normalization="max",
            label_preprocessing="one-hot",
            use_torch=True
        )

        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        transformations = [
            ToTensor(swap_color_axis=True),
            Normalize(mean=mean, std=std),
        ]

        if train and not validation:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
            transformations.append(RandomRotator(degrees=(0, 15)))

        cifar_data.set_transformation(transforms.Compose(transformations))

        if not train:
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(
                cifar_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return cifar_data, dataloader

        train_indices, validation_indices = self.get_validation_and_train_indices(cifar_data)
        indices = validation_indices if validation else train_indices
        subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)

        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        dataloader = DataLoader(
            cifar_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return cifar_data, dataloader

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """
        Creates the train/validation split from the CIFAR training set.
        """
        num_val_samples_per_class = self.num_val_samples_per_class
        num_train_samples_per_class = self.num_train_samples_per_class

        validation_set_size = self.num_classes * num_val_samples_per_class
        train_set_size = self.num_classes * num_train_samples_per_class

        validation_indices = torch.zeros(validation_set_size, dtype=torch.int64)
        train_indices = torch.zeros(train_set_size, dtype=torch.int64)

        current_val_samples = 0
        current_train_samples = 0

        for i in range(self.num_classes):
            class_indices = torch.argwhere(cifar_data.data["labels"][:, i] == 1).flatten()

            needed = num_val_samples_per_class + num_train_samples_per_class
            if class_indices.numel() < needed:
                raise ValueError(
                    f"Class {i} has only {class_indices.numel()} samples, but {needed} are required."
                )

            validation_indices[
                current_val_samples:current_val_samples + num_val_samples_per_class
            ] = class_indices[:num_val_samples_per_class]

            train_indices[
                current_train_samples:current_train_samples + num_train_samples_per_class
            ] = class_indices[
                num_val_samples_per_class:num_val_samples_per_class + num_train_samples_per_class
            ]

            current_val_samples += num_val_samples_per_class
            current_train_samples += num_train_samples_per_class

        return train_indices, validation_indices

    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader, val_dataloader: DataLoader,
              test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])
        self._save_model_parameters()

        for e in tqdm(range(self.current_epoch, self.num_epochs), desc="Epochs", dynamic_ncols=True):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()

            epoch_iterator = tqdm(
                enumerate(train_dataloader),
                total=len(train_dataloader),
                desc=f"Epoch {e+1}/{self.num_epochs}",
                leave=False,
                dynamic_ncols=True
            )

            for step_number, sample in epoch_iterator:
                image = sample["image"].to(self.device)
                label = sample["label"].to(self.device)

                self.optim.zero_grad(set_to_none=True)

                current_features = [] if self.use_cbp else None
                predictions = self.net.forward(image, current_features)[:, self.all_classes[:self.current_num_classes]]
                current_reg_loss = self.loss(predictions, label)
                current_loss = current_reg_loss.detach().clone()

                current_reg_loss.backward()
                self.optim.step()

                if self.use_cbp:
                    self.resgnt.gen_and_test(current_features)

                self.inject_noise()

                current_accuracy = torch.mean((predictions.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
                self.running_loss += current_loss
                self.running_accuracy += current_accuracy.detach()

                epoch_iterator.set_postfix(
                    loss=float(current_loss.item()),
                    acc=float(current_accuracy.item())
                )

                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

                    epoch_end_time = time.perf_counter()
                    self._store_test_summaries(
                        test_dataloader,
                        val_dataloader,
                        epoch_number=e,
                        epoch_runtime=epoch_end_time - epoch_start_time
                    )

                    self.current_epoch += 1
                    self.extend_classes(training_data, test_data, val_data)

                    if self.current_epoch % self.checkpoint_save_frequency == 0:
                        self.save_experiment_checkpoint()

    def set_lr(self):
        """
        Changes the learning rate of the optimizer according to the current epoch of the task.
        """
        current_stepsize = None
        task_epoch = self.current_epoch % self.class_increase_frequency

        if task_epoch == 0:
            current_stepsize = self.stepsize
        elif task_epoch == 60:
            current_stepsize = round(self.stepsize * 0.2, 5)
        elif task_epoch == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 5)
        elif task_epoch == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 5)

        if current_stepsize is not None:
            for g in self.optim.param_groups:
                g["lr"] = current_stepsize
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def inject_noise(self):
        """
        Adds a small amount of random noise to the parameters of the network.
        """
        if not self.perturb_weights_indicator:
            return

        with torch.no_grad():
            for param in self.net.parameters():
                param.add_(torch.randn(param.size(), device=param.device) * self.noise_std)

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet):
        """
        Adds new classes to the data set at a fixed frequency.
        """
        if (self.current_epoch % self.class_increase_frequency) == 0:
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))

            if self.early_stopping and len(self.best_accuracy_model_parameters) > 0:
                self.net.load_state_dict(self.best_accuracy_model_parameters)

            self.best_accuracy = torch.zeros_like(self.best_accuracy)
            self.best_accuracy_model_parameters = {}
            self._save_model_parameters()

            if self.current_num_classes == self.num_classes:
                return

            self.current_num_classes = min(self.current_num_classes + self.class_increment, self.num_classes)

            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew classes added... current_num_classes={0}".format(self.current_num_classes))

            if self.reset_head:
                kaiming_init_resnet_module(self.net.fc)
            if self.reset_network:
                self.net.apply(kaiming_init_resnet_module)

    def _save_model_parameters(self):
        """
        Stores the parameters of the model so it can be evaluated after the experiment is over.
        """
        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pt".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        store_object_with_several_attempts(
            self.net.state_dict(),
            file_path,
            storing_format="torch",
            num_attempts=10
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        action="store",
        type=str,
        default='./incremental_cifar/cfg/base_deep_learning_system.json',
        help="Path to the file containing the parameters for the experiment."
    )
    parser.add_argument(
        "--experiment-index",
        action="store",
        type=int,
        default=0,
        help="Index for the run; this will determine the random seed and the name of the results."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print extra information about the experiment as it's running."
    )

    # training setup
    parser.add_argument("--num-epochs", type=int, default=None, help="Total number of training epochs.")
    parser.add_argument("--num-classes", type=int, default=None, help="Total number of classes.")
    parser.add_argument("--initial-num-classes", type=int, default=None, help="Number of classes in the first task.")
    parser.add_argument("--class-increment", type=int, default=None, help="How many classes to add per task.")
    parser.add_argument("--class-increase-frequency", type=int, default=None, help="Epochs per task before adding new classes.")

    # dataloader / split setup
    parser.add_argument("--train-batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--validation-batch-size", type=int, default=None, help="Validation batch size.")
    parser.add_argument("--test-batch-size", type=int, default=None, help="Test batch size.")
    parser.add_argument("--num-train-samples-per-class", type=int, default=None, help="Training samples per class.")
    parser.add_argument("--num-val-samples-per-class", type=int, default=None, help="Validation samples per class.")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of dataloader workers.")

    # optimizer / experiment extras
    parser.add_argument("--stepsize", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=None, help="SGD momentum.")
    parser.add_argument("--running-avg-window", type=int, default=None, help="Window for online training summaries.")

    # optional toggles
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--reset-head", action="store_true", help="Reset classifier head at task boundaries.")
    parser.add_argument("--reset-network", action="store_true", help="Reset full network at task boundaries.")
    parser.add_argument("--early-stopping", action="store_true", help="Use best validation model within each task.")

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        experiment_parameters = json.load(config_file)

    cli_overrides = {
        "num_epochs": args.num_epochs,
        "num_classes": args.num_classes,
        "initial_num_classes": args.initial_num_classes,
        "class_increment": args.class_increment,
        "class_increase_frequency": args.class_increase_frequency,
        "train_batch_size": args.train_batch_size,
        "validation_batch_size": args.validation_batch_size,
        "test_batch_size": args.test_batch_size,
        "num_train_samples_per_class": args.num_train_samples_per_class,
        "num_val_samples_per_class": args.num_val_samples_per_class,
        "num_workers": args.num_workers,
        "stepsize": args.stepsize,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "running_avg_window": args.running_avg_window,
    }

    for key, value in cli_overrides.items():
        if value is not None:
            experiment_parameters[key] = value

    if args.debug:
        experiment_parameters["debug"] = True
    if args.reset_head:
        experiment_parameters["reset_head"] = True
    if args.reset_network:
        experiment_parameters["reset_network"] = True
    if args.early_stopping:
        experiment_parameters["early_stopping"] = True

    file_path = os.path.dirname(os.path.abspath(__file__))
    if "data_path" not in experiment_parameters or experiment_parameters["data_path"] == "":
        experiment_parameters["data_path"] = os.path.join(file_path, "data")
    if "results_dir" not in experiment_parameters or experiment_parameters["results_dir"] == "":
        experiment_parameters["results_dir"] = os.path.join(file_path, "results")
    if "experiment_name" not in experiment_parameters or experiment_parameters["experiment_name"] == "":
        experiment_parameters["experiment_name"] = "cifar_cbp_compare"

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(
        experiment_parameters,
        results_dir=os.path.join(experiment_parameters["results_dir"], experiment_parameters["experiment_name"]),
        run_index=args.experiment_index,
        verbose=args.verbose
    )
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()