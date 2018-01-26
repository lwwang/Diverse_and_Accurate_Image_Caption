
import tensorflow as tf
import numpy as np
import pdb


class Decoder:
    def __init__(self, image_embed_size, latent_size, word_embed_size, rnn_hidden_size, num_rnn_steps, vocab_size, attribute_embed_size):
        self.image_embed_size = image_embed_size
        self.latent_size = latent_size
        self.attribute_embed_size = attribute_embed_size
        self.word_embed_size = word_embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_steps = num_rnn_steps
        self.vocab_size = vocab_size
        self.rnn = tf.contrib.rnn.LSTMCell(self.rnn_hidden_size, state_is_tuple=False)

        self._initialize_weights()

    def _initialize_weights(self):
        self.weights = dict()
        self.biases = dict()

        self.weights['image_embed_to_hidden'] = tf.Variable(tf.random_uniform([self.image_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['image_embed_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))

        self.weights['latent_to_hidden'] = tf.Variable(tf.random_uniform([self.latent_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['latent_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))

        self.weights['attribute_to_hidden'] = tf.Variable(tf.random_uniform([self.attribute_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['attribute_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))

        self.weights['word_embeds'] = tf.Variable(tf.random_uniform([self.vocab_size, self.word_embed_size], -0.1, 0.1))
        self.weights['word_embed_to_hidden'] = tf.Variable(tf.random_uniform([self.word_embed_size, self.rnn_hidden_size], -0.1, 0.1))
        self.biases['word_embed_to_hidden'] = tf.Variable(tf.zeros([self.rnn_hidden_size]))
        self.weights['hidden_to_word'] = tf.Variable(tf.random_uniform([self.rnn_hidden_size, self.vocab_size], -0.1, 0.1))
        self.biases['hidden_to_word'] = tf.Variable(tf.zeros([self.vocab_size]))


    def decode(self, image, latent, sentence, mask, attribute):#, sentence2, mask3, max_num_questions):
        image_embeds = tf.matmul(image, self.weights['image_embed_to_hidden']) + self.biases['image_embed_to_hidden']
        z_embeds = tf.matmul(latent, self.weights['latent_to_hidden']) + self.biases['latent_to_hidden']
        attribute_embeds = tf.matmul(attribute, self.weights['attribute_to_hidden']) + self.biases['attribute_to_hidden']

        loss = tf.zeros([tf.shape(image)[0]])
        state = self.rnn.zero_state(tf.shape(image)[0], tf.float32)
        with tf.variable_scope('RNN_Decoder') as scope:
            for step in range(-3, self.num_rnn_steps):
                if step == -3:
                    curr_embeds = image_embeds
                elif step == -2:
                    curr_embeds = attribute_embeds
                elif step == -1:
                    curr_embeds = z_embeds
                else:
                    curr_embeds = tf.matmul(tf.nn.embedding_lookup(self.weights['word_embeds'], sentence[:, step]), self.weights['word_embed_to_hidden']) + \
                                  self.biases['word_embed_to_hidden']

                if step > -3: scope.reuse_variables()

                hidden, state = self.rnn(curr_embeds, state)

                if step == -3 or step == -2 or step == -1:
                    continue

                labels = tf.expand_dims(sentence[:, step+1], 1)
                indices = tf.expand_dims(tf.range(0, tf.shape(image)[0], 1), 1)
                concat = tf.concat([indices, labels], 1)
                onehot_labels = tf.sparse_to_dense(concat, tf.stack([tf.shape(image)[0], self.vocab_size]), 1.0, 0.0)

                logits = tf.matmul(hidden, self.weights['hidden_to_word']) + self.biases['hidden_to_word']
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits) * mask[:, step]
                loss = loss + cross_entropy

        return loss

    def decode_test(self):
        image = tf.placeholder(tf.float32, [None, self.image_embed_size])
        latent = tf.placeholder(tf.float32, [None, self.latent_size])
        attribute = tf.placeholder(tf.float32, [None, self.attribute_embed_size])

        batch_size = tf.shape(image)[0]

        question = []
        state = self.rnn.zero_state(batch_size, tf.float32)
        with tf.variable_scope('RNN_Decoder') as scope:
            for step in range(-3, self.num_rnn_steps - 1):
                scope.reuse_variables()

                if step == -3:
                    curr_embeds = tf.matmul(image,
                                            self.weights['image_embed_to_hidden']) + self.biases[
                                      'image_embed_to_hidden']

                if step == -2:
                    curr_embeds = tf.matmul(attribute,
                                            self.weights['attribute_to_hidden']) + self.biases[
                                      'attribute_to_hidden']

                if step == -1:

                    curr_embeds = tf.matmul(latent, self.weights['latent_to_hidden']) + self.biases['latent_to_hidden']

                if step == 0:
                    curr_embeds = tf.matmul(tf.nn.embedding_lookup(self.weights['word_embeds'],
                                                                   tf.zeros([tf.shape(image)[0]], dtype=tf.int32)),
                                            self.weights['word_embed_to_hidden']) + self.biases['word_embed_to_hidden']

                hidden, state = self.rnn(curr_embeds, state)

                if step == -3 or step == -2 or step == -1:
                    continue

                logits = tf.matmul(hidden, self.weights['hidden_to_word']) + self.biases['hidden_to_word']
                max_prob_word = tf.argmax(logits, dimension=1)

                curr_embeds = tf.matmul(tf.nn.embedding_lookup(self.weights['word_embeds'], max_prob_word),
                                        self.weights['word_embed_to_hidden']) + self.biases['word_embed_to_hidden']
                question.append(max_prob_word)

        return image, latent, attribute, tf.stack(question, axis=1)

    def decode_beam_initial(self):

        image = tf.placeholder(tf.float32, [None, self.image_embed_size])
        
        latent = tf.placeholder(tf.float32, [None, self.latent_size])

        attribute = tf.placeholder(tf.float32, [None, self.attribute_embed_size])

        state = self.rnn.zero_state(1, tf.float32)

        with tf.variable_scope('RNN_Decoder') as scope:

            for step in range(-3, 1):

                scope.reuse_variables()

                if step == -3:

                    curr_embeds = tf.matmul(tf.tile(image,[1, 1]), self.weights['image_embed_to_hidden']) + self.biases['image_embed_to_hidden']
                    hidden, state = self.rnn(curr_embeds, state)
                    continue

                if step == -2:

                    curr_embeds = tf.matmul(attribute, self.weights['attribute_to_hidden']) + self.biases['attribute_to_hidden']
                    hidden, state = self.rnn(curr_embeds, state)
                    continue

                if step == -1:

                    curr_embeds = tf.matmul(latent, self.weights['latent_to_hidden']) + self.biases['latent_to_hidden']
                    hidden, state = self.rnn(curr_embeds, state)
                    continue

                if step == 0:

                    curr_embeds = tf.matmul(
                        tf.nn.embedding_lookup(self.weights['word_embeds'], tf.zeros([1], dtype=tf.int32)),
                        self.weights['word_embed_to_hidden']) + self.biases['word_embed_to_hidden']

                    hidden, state = self.rnn(curr_embeds, state)
                    
                    logits = tf.matmul(hidden, self.weights['hidden_to_word']) + self.biases['hidden_to_word']

        return image, attribute, latent, logits, state


    def decode_beam_continue(self):

        state_last = tf.placeholder(tf.float32, [1, 2*self.rnn_hidden_size])

        word_idx = tf.placeholder(tf.int32, [1]) 

        with tf.variable_scope('RNN_Decoder') as scope:

            scope.reuse_variables()


            curr_embeds = tf.matmul(
                tf.nn.embedding_lookup(self.weights['word_embeds'], word_idx),
                self.weights['word_embed_to_hidden']) + self.biases['word_embed_to_hidden']

            hidden, state_current = self.rnn(curr_embeds, state_last)

            logit_curr = tf.matmul(hidden, self.weights['hidden_to_word']) + self.biases['hidden_to_word']


        return word_idx, state_last, state_current, logit_curr


    def extract_visualFea(self):

        image = tf.placeholder(tf.float32, [None, self.image_embed_size])
        curr_embeds = tf.matmul(image, self.weights['image_embed_to_hidden']) + self.biases['image_embed_to_hidden']
        imageFea = curr_embeds
        return image, imageFea





