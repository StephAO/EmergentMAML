from agent import Agent
import tensorflow as tf
import tensorflow_probability as tfp

class SenderAgent(Agent):

    def __init__(self):
        super().__init__()
        self.num_fc_hidden = 256
        # TODO define temperature better - maybe learnt temperature?
        self.temperature = 0.2

    def _build_input(self):
        """
        Define starting state and inputs to next state
        For sender agent:
            - starting state is output of a pre-trained model on imagenet
            - First input is a start of sentence token (sos), followed by the output of the previous timestep
        :return:
        """
        # Determines starting state
        self.target_image = tf.placeholder(tf.float32)
        self.s0 = self.pre_trained(self.target_image)

        # Determines input to decoder at next time step
        self.helper = tf.contrib.seq2seq.InferenceHelper(sample_fn=lambda outputs: outputs,
                                                         sample_shape=[self.input_size],
                                                         sample_dtype=tf.dtypes.float32,
                                                         start_inputs=self.sos_token,
                                                         end_fn=lambda sample_ids: tf.equal(sample_ids, self.eos_token))


    def _build_output(self):
        super()._build_output()
        self.final_features = tf.layers.dense(self.output, self.num_fc_hidden, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.final_features, self.vocab_size, activation=None)
        self.output = tfp.distributions.RelaxedOneHotCategorical(self.temperature, logits=self.logits)




