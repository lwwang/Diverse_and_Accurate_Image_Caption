import tensorflow as tf
import pdb

class Encoder:
    def __init__(self, image_embed_size, word_embed_size, rnn_hidden_size, num_rnn_steps, latent_size, vocab_size, cluster_embed_size):
        self.image_embed_size = image_embed_size
        self.word_embed_size = word_embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_steps = num_rnn_steps
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.cluster_embed_size = cluster_embed_size
        self.rnn = tf.contrib.rnn.LSTMCell(self.rnn_hidden_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        self.weights = dict()
        self.biases = dict()

        self.weights['image_embed_to_hidden'] = tf.Variable(tf.random_uniform([self.image_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['image_embed_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))
        self.weights['word_embeds'] = tf.Variable(tf.random_uniform([self.vocab_size, self.word_embed_size], -0.1, 0.1))
        self.weights['word_embed_to_hidden'] = tf.Variable(tf.random_uniform([self.word_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['word_embed_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))
        self.weights['cluster_embed_to_hidden'] = tf.Variable(tf.random_uniform([self.cluster_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['cluster_embed_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))
        self.weights['hidden_to_gaussian_params'] = tf.Variable(tf.random_uniform([self.rnn_hidden_size, 2 * self.latent_size * self.cluster_embed_size], -0.1, 0.1))
        self.biases['hidden_to_gaussian_params'] = tf.Variable(tf.zeros([2 * self.latent_size * self.cluster_embed_size]))

    def encode(self, image, sentence, mask2, cluster_vec):
        image_embeds = tf.matmul(image, self.weights['image_embed_to_hidden']) + self.biases['image_embed_to_hidden']
        cluster_embeds = tf.matmul(cluster_vec, self.weights['cluster_embed_to_hidden']) + self.biases['cluster_embed_to_hidden']
        
        joint_embeds = tf.zeros([tf.shape(image)[0], self.rnn_hidden_size])

        state = self.rnn.zero_state(tf.shape(image)[0], tf.float32)

        with tf.variable_scope('RNN_Encoder') as scope:
            for step in range(self.num_rnn_steps):
                if step == 0:
                    curr_embeds = image_embeds
                else:                    
                    if step == 1:
                        curr_embeds = cluster_embeds
                    else:
                        curr_embeds = tf.matmul(tf.nn.embedding_lookup(self.weights['word_embeds'], sentence[:, step]), \
                            self.weights['word_embed_to_hidden']) + self.biases['word_embed_to_hidden']

                if step > 0:
                    scope.reuse_variables()

                hidden, state = self.rnn(curr_embeds, state)
                joint_embeds += hidden * tf.tile(tf.slice(mask2, [0, step], [tf.shape(image)[0], 1]), [1, self.rnn_hidden_size])

        
        mu = tf.matmul(joint_embeds, self.weights['hidden_to_gaussian_params'][:, (0*2)*self.latent_size:(0*2+1)*self.latent_size]) \
                + self.biases['hidden_to_gaussian_params'][(0*2)*self.latent_size:(0*2+1)*self.latent_size]

        log_sigma_sq = tf.matmul(joint_embeds, self.weights['hidden_to_gaussian_params'][:, (0*2+1)*self.latent_size:(0*2+2)*self.latent_size]) \
                + self.biases['hidden_to_gaussian_params'][(0*2+1)*self.latent_size:(0*2+2)*self.latent_size]

        epsilon = tf.random_normal([tf.shape(image)[0], self.latent_size])

        z = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), epsilon))

        for idx_cluster in range(self.cluster_embed_size-1):

            idx_cluster += 1

            mu_idx = tf.matmul(joint_embeds, self.weights['hidden_to_gaussian_params'][:, (idx_cluster*2)*self.latent_size:(idx_cluster*2+1)*self.latent_size]) \
                + self.biases['hidden_to_gaussian_params'][(idx_cluster*2)*self.latent_size:(idx_cluster*2+1)*self.latent_size]

            mu = tf.concat([mu, mu_idx], 1)

            log_sigma_sq_idx = tf.matmul(joint_embeds, self.weights['hidden_to_gaussian_params'][:, (idx_cluster*2+1)*self.latent_size:(idx_cluster*2+2)*self.latent_size]) \
                + self.biases['hidden_to_gaussian_params'][(idx_cluster*2+1)*self.latent_size:(idx_cluster*2+2)*self.latent_size]

            log_sigma_sq = tf.concat([log_sigma_sq, log_sigma_sq_idx], 1)

            epsilon_idx = tf.random_normal([tf.shape(image)[0], self.latent_size])

            z_idx= tf.add(mu_idx, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq_idx)), epsilon_idx))

            z = tf.concat([z, z_idx], 1)

        return z, mu, log_sigma_sq

