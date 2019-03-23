"""
Based heavily on https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/reptile.py
"""
from comet_ml import Experiment
from Tasks import image_captioning as ic, referential_game as rg
from utils.variables import *
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner, ImageSelector
from Tasks import ReferentialGame, ImageCaptioning, ImageSelection
from utils.data_handler import Data_Handler
import tensorflow as tf

"""
Steps: 
1) Create necessary functions to easily access different sets of variables - DONE
2) Create optimizer for each task so that only relevant variables are updates - DONE
3) Deal with how data will be distributed across tasks - DONE
4) Use above to write reptile
5) Extensive testing
"""


class Reptile:
    """
    A meta-learning session.
    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, data_handler, sender=True, receiver=True, image_captioner=True, image_selector=True, track_results=True):
        self.sess = Agent.sess
        self.N = 6 # number of steps taken for each task - should be > 1

        self.S = SenderAgent()
        self.R = ReceiverAgent(*self.S.get_output())
        self.IC = ImageCaptioner()
        self.IS = ImageSelector()
        self.step = 1

        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                     project_name='Reptile',
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))

        self.T = {}
        if image_captioner:
            self.ic = ImageCaptioning(self.IC, experiment=self.experiment, track_results=False)
            self.T["Image Captioner"] = lambda img, capts: self.ic.train_batch(images=img, captions=capts, mode="train")
        if image_selector:
            self.is_ = ImageSelection(self.IS, experiment=self.experiment, track_results=False)
            self.T["Image Selector"] = lambda img, capts: self.is_.train_batch(images=img, captions=capts, mode="train")
        if sender or receiver:
            self.rg = ReferentialGame(self.S, self.R, experiment=self.experiment, track_results=False)
            if receiver:
                self.T["Receiver"] = lambda img, capts: self.rg.train_batch(images=img, mode="receiver_train")
            if sender:
                self.T["Sender"] = lambda img, capts: self.rg.train_batch(images=img, mode="sender_train")


        self.maml_state = VariableState(self.sess, Agent.get_weights())
        self.sender_state = VariableState(self.sess, SenderAgent.get_weights())
        self.receiver_state = VariableState(self.sess, ReceiverAgent.get_weights())

        self.states = {"maml": self.maml_state, "s": self.sender_state, "r": self.receiver_state}

        self.dh = data_handler

    def get_experiment_key(self):
        return self.experiment.get_key()

    def get_diff(self, a, b):
        diff = 0.
        if isinstance(a, (np.ndarray, np.generic)):
            return np.sum(np.abs(a - b))

        elif isinstance(a, list):
            for i in range(len(a)):
                diff += self.get_diff(a[i], b[i])

        elif isinstance(a, dict):
            for k in a:
                diff += self.get_diff(a[k], b[k])

        return diff

    def train_epoch(self, e, mode=None):
        image_gen = self.dh.get_images(return_captions=True, mode="train")
        self.experiment.set_step(self.step)
        start_vars = {k: s.export_variables() for k, s in self.states.items()}

        while True:
            try:
                # Save current variables
                old_vars = {k: s.export_variables() for k, s in self.states.items()}
                new_vars = {k: [] for k, s in self.states.items()}

                # For each task
                for task in ["Image Captioner", "Image Selector", "Sender", "Receiver"]:
                    # parameter setup to not waste data
                    if task in ["Sender", "Receiver", "Image Selector"]:
                        self.dh.set_params(images_per_instance=Agent.D+1)
                    else:
                        self.dh.set_params(images_per_instance=1)
                    # Run task n times
                    for _ in range(self.N):
                        images, captions = next(image_gen)
                        acc, loss = self.T[task](images, captions)

                    self.train_metrics[task + " Accuracy"] = acc
                    self.train_metrics[task + " Loss"] = loss
                    # Store new variables
                    [new_vars[k].append(s.export_variables()) for k, s in self.states.items()]
                    # Reset to old variables for next task
                    [s.import_variables(old_vars[k]) for k, s in self.states.items()]
                self.experiment.log_metrics(self.train_metrics)
                self.step += 1
                self.experiment.set_step(self.step)
                # Average new variables
                new_vars = {k: average_vars(new_vars[k]) for k, s in self.states.items()}
                # Set variables to new variables
                [s.import_variables(interpolate_vars(old_vars[k], new_vars[k], 0.2)) for k, s in self.states.items()]

            except StopIteration:
                break
        end_vars = {k: s.export_variables() for k, s in self.states.items()}
        weight_diff = self.get_diff(start_vars, end_vars)

        #self.experiment.set_step(e)
        self.val_metrics["Weight Change"] = weight_diff
        self.experiment.log_metrics(self.val_metrics)

        return 0, weight_diff

if __name__ == "__main__":
    a = SenderAgent(5, 10, 15)
    r = Reptile(a, None)
