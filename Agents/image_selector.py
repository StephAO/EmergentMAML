import tensorflow as tf

from Agents.receiver_agent import ReceiverAgent
from Agents.agent import Agent

class ImageSelector(ReceiverAgent):
    """
    A class for implementing an agent that selects between a set of images
    based on a given caption
    """
    
    def __init__(self, vocab_size, num_distractors, max_len, loss_type='pairwise', **kwargs):
        # TODO - Not sure what is the difference betwenn message_len and max_len
        super().__init__(vocab_size, num_distractors, max_len, 
            tf.placeholder(dtype=tf.float32, shape=(None, Agent.batch_size, vocab_size)),
            0, loss_type=loss_type, **kwargs)
                
    def get_output(self):
        return self.prediction
        
    def fill_feed_dict(self, fd, captions, candidates, target_idx):
        fd[self.message] = captions
        super().fill_feed_dict(fd, candidates, target_idx)
    
    