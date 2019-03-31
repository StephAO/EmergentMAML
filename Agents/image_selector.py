import tensorflow as tf

from Agents.receiver_agent import ReceiverAgent
from Agents.agent import Agent

class ImageSelector(ReceiverAgent):
    """
    A class for implementing an agent that selects between a set of images
    based on a given caption
    """
    
    def __init__(self, **kwargs):
        # TODO - Not sure what is the difference betwenn message_len and max_len
        message = tf.placeholder(dtype=tf.int32, shape=(Agent.batch_size, Agent.L, Agent.K))
        super().__init__(message, Agent.L, **kwargs)


    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=Agent.step,
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip,
            # only update image selector weights
            variables=ImageSelector.get_all_weights()
        )

    def get_train_ops(self):
        return [self.train_op]
        
    def fill_feed_dict(self, fd, captions, candidates, target_idx):
        fd[self.message] = captions
        super().fill_feed_dict(fd, candidates, target_idx)
    
    