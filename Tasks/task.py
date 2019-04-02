from comet_ml import Experiment
import numpy as np
from Agents.agent import Agent

class Task(object):
    """
    Abstract Base Task class
    """
    def __init__(self, agents, data_handler=None, experiment=None, track_results=True):
        """
        Initialize this Image Captioning task
        """
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        # get params from agent
        self.L = Agent.L
        self.K = Agent.K
        self.V = Agent.V
        self.D = Agent.D
        self.batch_size = Agent.batch_size

        self.dh = data_handler

        # Get necessary ops to run
        self.run_ops = list(self.agents[-1].get_output())
        self.train_ops = self.agents[-1].get_train_ops()

        self.track_results = track_results
        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = experiment or Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                                   project_name=self.name,
                                                   auto_param_logging=False, auto_metric_logging=False,
                                                   disabled=(not track_results))
        if experiment is None:
            self.params = {}
            self.params.update(Agent.get_params())
            self.params.update(self.dh.get_params())
            self.experiment.log_parameters(self.params)

        self.captions_required = False

    def get_experiment_key(self):
        return self.experiment.get_key()

    def train_epoch(self, e, mode="train"):
        """
        Play an epoch of a task defined by iterating over the dataset once
        """
        losses = []
        accuracies = []

        image_gen = self.dh.get_images(return_captions=self.captions_required, mode=mode)
        while True:
            try:
                next_inputs = next(image_gen)
                acc, loss = self.train_batch(next_inputs, mode=mode)
                losses.append(loss)
                accuracies.append(acc)
            except StopIteration:
                break

        avg_acc, avg_loss  = np.mean(accuracies), np.mean(losses)
        if self.track_results:
            self.experiment.set_step(e)
            metrics = self.train_metrics if mode == "train" else self.val_metrics
            mode_name = " Training " if mode == "train" else " Validation "
            metrics[self.name + mode_name + "Accuracy"] = avg_acc
            metrics[self.name + mode_name + "Loss"] = avg_loss
            self.experiment.log_metrics(metrics)

        return avg_acc, avg_loss

    def train_batch(self, inputs, mode="train"):
        pass

    def run_game(self, fd, mode="train"):
        ops = self.run_ops
        if mode == "train":
            ops += self.train_ops
        elif mode == "sender_train":
            ops += [self.train_ops[0]]
        elif mode == "receiver_train":
            ops += [self.train_ops[1]]
        acc, loss, prediction = Agent.sess.run(ops, feed_dict=fd)[:3]
        return acc, loss, prediction