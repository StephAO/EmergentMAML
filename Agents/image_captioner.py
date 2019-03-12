import tensorflow as tf

from Agents.sender_agent import SenderAgent


class ImageCaptioner(SenderAgent):

    def _build_input(self):
        """ 
        Build self's starting state using the pre-trained cnn on imagenet 
        """
        self.image_captions = tf.placeholder(tf.int32, shape=(self.batch_size, self.L))
        super()._build_input()
    
    def _build_losses(self):
        """ 
        Build this agent's loss function
        """

        # TODO - currently using the default softmax function - check for potentially something fancier
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.rnn_outputs,
            self.image_captions,
            tf.ones([self.batch_size, self.L], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.transpose(self.image_captions)), tf.float32))

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.epoch,
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip)

    def get_output(self):
        return self.output, self.prediction, self.final_sequence_lengths

    def fill_feed_dict(self, feed_dict, target_image, image_captions):
        feed_dict[self.target_image] = target_image
        feed_dict[self.image_captions] = image_captions
    
    def close(self):
        self.sess.close()

    def __del__(self):
        self.sess.close()
    
    def run_game(self, fd):
        return self.sess.run([self.train_op, self.accuracy, self.loss], feed_dict=fd)[1:]
