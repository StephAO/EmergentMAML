"""
Based heavily on https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/reptile.py
"""
from comet_ml import Experiment
from Tasks import image_captioning as ic, referential_game as rg
from utils.variables import *
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner, ImageSelector
from Tasks import Task, ReferentialGame, ImageCaptioning, ImageSelection
from utils.data_handler import Data_Handler, project_path
import tensorflow as tf

"""
Steps: 
1) Create necessary functions to easily access different sets of variables - DONE
2) Create optimizer for each task so that only relevant variables are updates - DONE
3) Deal with how data will be distributed across tasks - DONE
4) Use above to write reptile
5) Extensive testing
"""


class Reptile(Task):
    """
    A meta-learning task that teaches an agent over a set of other tasks
    """
    def __init__(self, data_handler, sender=True, receiver=True, image_captioner=True, image_selector=True, track_results=True):
        self.sess = Agent.sess
        self.N = 4 # number of steps taken for each task - should be > 1

        self.S = SenderAgent()
        self.R = ReceiverAgent(*self.S.get_output())
        self.IC = ImageCaptioner()
        self.IS = ImageSelector()

        self.train_metrics = {}
        self.val_metrics = {}
        self.experiment = Experiment(api_key='1jl4lQOnJsVdZR6oekS6WO5FI',
                                     project_name='Reptile',
                                     auto_param_logging=False, auto_metric_logging=False,
                                     disabled=(not track_results))

        self.params = {}
        self.params.update(Agent.get_params())
        self.params.update(data_handler.get_params())
        self.experiment.log_parameters(self.params)

        self.T = {}
        if image_captioner:
            self.ic = ImageCaptioning(self.IC, experiment=self.experiment, track_results=False)
            self.T["Image Captioner"] = lambda img, capts: self.ic.train_batch((img,capts), mode="train")
        if image_selector:
            self.is_ = ImageSelection(self.IS, experiment=self.experiment, track_results=False)
            self.T["Image Selector"] = lambda img, capts: self.is_.train_batch((img,capts), mode="train")
        if sender or receiver:
            self.rg = ReferentialGame(self.S, self.R, experiment=self.experiment, track_results=False)
            if receiver:
                self.T["Receiver"] = lambda img, capts: self.rg.train_batch(img, mode="receiver_train")
            if sender:
                self.T["Sender"] = lambda img, capts: self.rg.train_batch(img, mode="sender_train")


        self.sender_shared_state = VariableState(self.sess, SenderAgent.get_shared_weights())
        self.receiver_shared_state = VariableState(self.sess, ReceiverAgent.get_shared_weights())
        self.sender_own_state = VariableState(self.sess, SenderAgent.get_weights())
        self.receiver_own_state = VariableState(self.sess, ReceiverAgent.get_weights())

        # print(SenderAgent.get_shared_weights())
        # print(ReceiverAgent.get_shared_weights())
        # print(SenderAgent.get_weights())
        # print(ReceiverAgent.get_weights())
        # print(tf.trainable_variables())

        self.shared_states = {"shared_sender": self.sender_shared_state, "shared_receiver": self.receiver_shared_state}
        self.own_states = {"own_sender": self.sender_own_state, "own_receiver": self.receiver_own_state}

        variables_to_initialize = tf.global_variables()
        Agent.sess.run(tf.variables_initializer(variables_to_initialize))

        shared_average = []
        for k, v in self.shared_states.items():
            shared_average.append(v.export_variables())

        shared_average = np.mean(shared_average, axis=0)
        self.set_weights(new_shared_weights=shared_average)

        self.dh = data_handler
        with open("{}/data/csv_loss_{}.csv".format(project_path, self.experiment.get_key()), 'w+') as csv_loss_file:
            csv_loss_file.write("Image Captioner Loss,Image Selector Loss,Sender Loss,Receiver Loss\n")
        with open("{}/data/csv_accuracy_{}.csv".format(project_path, self.experiment.get_key()), 'w+') as csv_acc_file:
            csv_acc_file.write("Image Captioner Loss,Image Selector Loss,Sender Loss,Receiver Loss\n")

        self.step = 0


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

    def set_weights(self, new_own_weights=None, new_shared_weights=None):
        if new_own_weights is not None:
            for k, s in self.own_states.items():
                s.import_variables(new_own_weights[k])
        if new_shared_weights is not None:
            for k, s in self.shared_states.items():
                s.import_variables(new_shared_weights)

    def train_epoch(self, e, mode=None):
        self.dh.set_params(distractors=0)
        image_gen = self.dh.get_images(return_captions=True, mode="train")
        # Get current variables
        start_vars = {k: s.export_variables() for k, s in self.own_states.items()}
        start_vars["shared"] = self.shared_states["shared_sender"].export_variables()

        while True:
            try:

                # Save current variables
                old_own = {k: s.export_variables() for k, s in self.own_states.items()}
                new_own = {k: [] for k, s in self.own_states.items()}
                old_shared = self.shared_states["shared_sender"].export_variables()
                new_shared = []

                # For each task
                for task in ["Image Captioner", "Image Selector", "Sender", "Receiver"]:
                    # parameter setup to not waste data
                    if task in ["Sender", "Receiver", "Image Selector"]:
                        self.dh.set_params(distractors=Agent.D)
                    else:
                        self.dh.set_params(distractors=0)
                    # Run task n times
                    for _ in range(self.N):
                        images, captions = next(image_gen)
                        acc, loss = self.T[task](images, captions)
                    self.train_metrics[task + " Accuracy"] = acc
                    self.train_metrics[task + " Loss"] = loss

                    # Store new variables
                    [new_own[k].append(s.export_variables()) for k, s in self.own_states.items()]
                    [new_shared.append(s.export_variables()) for k, s in self.shared_states.items()]

                    # Reset to old variables for next task
                    [s.import_variables(old_own[k]) for k, s in self.own_states.items()]
                    [s.import_variables(old_shared) for k, s in self.shared_states.items()]

                self.step += 1
                self.experiment.set_step(self.step)
                self.experiment.log_metrics(self.train_metrics)
                # Average new variables
                new_own = {k: interpolate_vars(old_own[k], average_vars(new_own[k]), 0.1) for k, s in self.own_states.items()}
                new_shared = interpolate_vars(old_shared, average_vars(new_shared), 0.1)
                # Set variables to new variables
                self.set_weights(new_own_weights=new_own, new_shared_weights=new_shared)

            except StopIteration:
                break

        # Get change in weights
        end_vars = {k: s.export_variables() for k, s in self.own_states.items()}
        end_vars["shared"] = self.shared_states["shared_sender"].export_variables()
        weight_diff = self.get_diff(start_vars, end_vars)

        #self.experiment.set_step(e)
        self.val_metrics["Weight Change"] = weight_diff
        self.experiment.log_metrics(self.val_metrics)

        # Log data to a csv
        with open("{}/data/csv_loss_{}.csv".format(project_path, self.experiment.get_key()), 'a') as csv_loss_file, \
             open("{}/data/csv_accuracy_{}.csv".format(project_path, self.experiment.get_key()), 'a') as csv_acc_file:
            losses = []
            accs = []
            for task in ["Image Captioner", "Image Selector", "Sender", "Receiver"]:
                losses.append(str(self.train_metrics[task + " Loss"]))
                accs.append(str(self.train_metrics[task + " Accuracy"]))

            csv_loss_file.write(",".join(losses))
            csv_loss_file.write("\n")

            csv_acc_file.write(",".join(accs))
            csv_acc_file.write("\n")

        return 0, weight_diff

