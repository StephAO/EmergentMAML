from Agents.agent import Agent
from data_handler import img_h, img_w
import tensorflow as tf
import tensorflow_probability as tfp

class ImageCaptioner(Agent):
    
    def __init__(self, vocab_size, L=3):
        # TODO - wasn't sure what to set the # of distractors to --> move it out of agent?
        
        self.temperature = 5.
        self.caption_len = L
        super().__init__(vocab_size, 0)
        
    def _build_input(self):
        """ 
        Build self's starting state using the pre-trained cnn on imagenet 
        """
        self.target_image = \
            tf.placeholder(tf.float32, shape=(self.batch_size, img_h, img_w, 3))
            
        # Todo - Not sure how this will apply to variable sequence captions!
        self.image_caption = \
            tf.placeholder(tf.int32, shape=(self.batch_size, self.caption_len))
            
        self.image_fets = Agent.pre_trained(self.target_image)
        self.image_fets = tf.stop_gradient(self.image_fets)
        
        # Todo - check if we need to normalize the fets like we did for the sender
        
        self.s0 = Agent.img_fc(self.image_fets)
        
        self.starting_tokens = tf.stack([self.sos_token] * self.batch_size)
        
        # TODO - should use TrainingHelper during training and inference helper during learning??? --> check and apply if applicable
        self.helper = \
            tf.contrib.seq2seq.InferenceHelper(
                sample_fn=lambda outputs: outputs,
                sample_shape=[self.K],
                sample_dtype=tf.float32,
                start_inputs=self.starting_tokens,
                end_fn=lambda sample_ids:
                    tf.reduce_all(tf.equal(sample_ids, self.eos_token))
            )
            
    def _build_output(self):
        """
        Build self's output (sequence from the RNN)
        """
        output_to_input = tf.layers.Dense(
                self.K,
                kernel_initializer=tf.glorot_uniform_initializer)

        # Decoder
        self.decoder = tf.contrib.seq2seq.BasicDecoder(
            Agent.gru_cell, 
            self.helper, 
            initial_state=self.s0,
            output_layer=output_to_input)

        self.rnn_outputs, self.final_state, self.final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(
                self.decoder, 
                output_time_major=True, 
                maximum_iterations=self.caption_len)
        
        self.rnn_outputs = self.rnn_outputs[0]
        
        self.dist = tfp.distributions.RelaxedOneHotCategorical(
            self.temperature, 
            logits=self.rnn_outputs)
        
        self.output = self.dist.sample()
        
        self.output_symbol = tf.argmax(self.output, axis=2)
    
    def _build_losses(self):
        """ 
        Build this agent's loss function
        """
        
        # TODO - verify that this works for variable length captions and if not find an alternative
        
        # TODO - currently using the default softmax function - check for potentially something fancier
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.rnn_outputs,
            self.image_caption,
            tf.ones([self.batch_size, self.caption_len], dtype=tf.dtypes.float32),
            average_across_timesteps=True,
            average_across_batch=True)
    
    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.epoch,
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip)
    
    def get_output(self):
        return self.output, self.output_symbol, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image, image_caption):
        feed_dict[self.target_image] = target_image
        feed_dict[self.image_caption] = image_caption
    
    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()
    
    def run_game(self, fd):
        return self.sess.run([self.train_op, self.output_symbol, self.loss], feed_dict=fd)[1:]
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            