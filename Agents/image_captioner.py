import tensorflow as tf

from .agent import Agent
from .sender_agent import SenderAgent

class ImageCaptioner(SenderAgent):

    def set_embedding(self, embedding):
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=(Agent.K, Agent.emb_size))
        self.embedding_init = self.embedding.assign(self.embedding_placeholder)
        Agent.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})

    def _build_input(self):
        """ 
        Build self's starting state using the pre-trained cnn on imagenet 
        """
        self.in_captions = tf.placeholder(tf.int32, shape=(self.batch_size, self.L))
        self.out_captions = tf.placeholder(tf.int32, shape=(self.batch_size, self.L))

        # Define the variable that will hold the embedding:
        self.embedding = tf.Variable(tf.zeros((Agent.K, Agent.emb_size)), trainable=False, name="Embedding")

        self.in_embs = tf.nn.embedding_lookup(self.embedding, self.in_captions)

        super()._build_input()
        self.input = tf.concat((self.in_embs, self.pre_feat), axis=1)
        self.helper = tf.contrib.seq2seq.TrainingHelper(self.input, [self.L]*self.batch_size) #tf.one_hot(self.in_captions, self.K), [self.L]*self.batch_size)
    
    def _build_losses(self):
        """ 
        Build this agent's loss function
        """
        # TODO - should use TrainingHelper during training and inference helper during validation
        # Sender agent helper is inference helper
        # TODO - currently using the default softmax function - check for potentially something fancier
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.rnn_outputs,
            self.out_captions,
            tf.ones([self.batch_size, self.L], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.transpose(self.out_captions)), tf.float32))

    def _build_optimizer(self):
        self.train_op = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=Agent.step,
            learning_rate=self.lr,
            optimizer="Adam",
            # some gradient clipping stabilizes training in the beginning.
            clip_gradients=self.gradient_clip,
            # only update image captioner weights
            variables=ImageCaptioner.get_all_weights()
        )

    def get_output(self):
        return self.accuracy, self.loss, self.prediction

    def get_train_ops(self):
        return [self.train_op]

    def fill_feed_dict(self, feed_dict, target_image, in_captions, out_captions):
        feed_dict[self.target_image] = target_image
        feed_dict[self.in_captions] = in_captions
        feed_dict[self.out_captions] = out_captions
    
    def close(self):
        Agent.sess.close()

    def __del__(self):
        Agent.sess.close()
