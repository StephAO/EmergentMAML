"""
Based heavily on https://github.com/openai/supervised-reptile/blob/master/supervised_reptile/reptile.py
"""

from Tasks import image_captioning as ic, referential_game as rg
from utils.variables import *
from Agents import Agent, SenderAgent, ReceiverAgent, ImageCaptioner
from Tasks import ReferentialGame, ImageCaptioning
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
    def __init__(self, sender=True, receiver=True, image_captioner=True, variables=None, transductive=False, pre_step_op=None):
        self.sess = Agent.sess
        self.N = 4 # number of steps taken for each task - should be > 1

        self.S = SenderAgent()
        self.R = ReceiverAgent(*self.S.get_output())
        self.S.set_loss(self.R.get_output()[1])
        self.IC = ImageCaptioner()

        self.T = {}
        if sender or receiver:
            self.rg = ReferentialGame(self.S, self.R)
            if sender:
                self.T["s"] = lambda img, capts: self.rg.train_batch(images=img, mode="train_sender")
            if receiver:
                self.T["r"] = lambda img, capts: self.rg.train_batch(images=img, mode="train_receiver")
        if image_captioner:
            self.ic = ImageCaptioning(self.IC)
            self.T["ic"] = lambda img, capts: self.ic.train_batch(images=img, captions=capts, mode="train")

        self.maml_state = VariableState(self.sess, Agent.get_weights())
        self.sender_state = VariableState(self.sess, SenderAgent.get_weights())
        self.receiver_state = VariableState(self.sess, ReceiverAgent.get_weights())

        self.states = {"maml": self.maml_state, "s": self.sender_state, "r": self.receiver_state}

        self.dh = Data_Handler()



        # self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        # self._full_state = VariableState(self.session,
        #                                  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        # self._transductive = transductive
        # self._pre_step_op = pre_step_op

    def train_epoch(self):

        losses = []
        accuracies = []

        image_gen = self.dh.get_images(return_captions=True, mode="train")
        while True:
            try:
                # Save current variables
                old_vars = {k: s.export_variables() for k, s in self.states.items()}
                new_vars = {k: [] for k, s in self.states.items()}

                # For each task
                for tasks in self.T:
                    # parameter setup to not waste data
                    if tasks in ["s", "r"]:
                        self.dh.set_params(images_per_instance=Agent.D+1)
                    else:
                        self.dh.set_params(images_per_instance=1)
                    # Run task n times
                    for _ in range(self.N):
                        images, captions = next(image_gen)
                        acc, loss = self.T[tasks](images, captions)

                    # Store new variables
                    [new_vars[k].append(s.export_variables()) for k, s in self.states.items()]
                    # Reset to old variables for next task
                    [s.import_variables(old_vars[k]) for k, s in self.states.items()]

                # Average new variables
                new_vars = {average_vars(new_vars[k]) for k, s in self.states.items()}
                # Set variables to new variables
                [s.import_variables(interpolate_vars(old_vars[k], new_vars[k], 0.1)) for k, s in self.states.items()]

            except StopIteration:
                break


    # def evaluate(self,
    #              dataset,
    #              input_ph,
    #              label_ph,
    #              minimize_op,
    #              predictions,
    #              num_classes,
    #              num_shots,
    #              inner_batch_size,
    #              inner_iters,
    #              replacement):
    #     """
    #     Run a single evaluation of the model.
    #     Samples a few-shot learning task and measures
    #     performance.
    #     Args:
    #       dataset: a sequence of data classes, where each data
    #         class has a sample(n) method.
    #       input_ph: placeholder for a batch of samples.
    #       label_ph: placeholder for a batch of labels.
    #       minimize_op: TensorFlow Op to minimize a loss on the
    #         batch specified by input_ph and label_ph.
    #       predictions: a Tensor of integer label predictions.
    #       num_classes: number of data classes to sample.
    #       num_shots: number of examples per data class.
    #       inner_batch_size: batch size for every inner-loop
    #         training iteration.
    #       inner_iters: number of inner-loop iterations.
    #       replacement: sample with replacement.
    #     Returns:
    #       The number of correctly predicted samples.
    #         This always ranges from 0 to num_classes.
    #     """
    #     train_set, test_set = _split_train_test(
    #         _sample_mini_dataset(dataset, num_classes, num_shots+1))
    #     old_vars = self._full_state.export_variables()
    #     for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
    #         inputs, labels = zip(*batch)
    #         if self._pre_step_op:
    #             self.session.run(self._pre_step_op)
    #         self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
    #     test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
    #     num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
    #     self._full_state.import_variables(old_vars)
    #     return num_correct
    #
    # def _test_predictions(self, train_set, test_set, input_ph, predictions):
    #     if self._transductive:
    #         inputs, _ = zip(*test_set)
    #         return self.session.run(predictions, feed_dict={input_ph: inputs})
    #     res = []
    #     for test_sample in test_set:
    #         inputs, _ = zip(*train_set)
    #         inputs += (test_sample[0],)
    #         res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
    #     return res


if __name__ == "__main__":
    a = SenderAgent(5, 10, 15)
    r = Reptile(a, None)