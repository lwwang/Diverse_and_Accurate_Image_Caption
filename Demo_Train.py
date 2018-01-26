# Liwei Wang
# Implementation of AG-CVAE for image captioning

import tensorflow as tf
tf.set_random_seed(456)
import h5py, json, argparse, string
import numpy as np
np.random.seed(123)
import pickle, glob, os, nltk
from Encoder import *
from Decoder import *
import subprocess
from scipy.misc import imread, imresize, imsave, imshow
import pdb
import sys
import copy
import timeit

class VariationalAutoencoder:
    def __init__(self, image_embed_size, word_embed_size, rnn_hidden_size, num_rnn_steps, vocab_size, latent_size, cluster_embed_size):
        self.encoder = Encoder(image_embed_size, word_embed_size, rnn_hidden_size, num_rnn_steps,
                               latent_size, vocab_size, cluster_embed_size)
        self.decoder = Decoder(image_embed_size, latent_size, word_embed_size, rnn_hidden_size, num_rnn_steps,
                               vocab_size, cluster_embed_size)

        self.image_embed_size = image_embed_size
        self.latent_size = latent_size
        self.word_embed_size = word_embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_steps = num_rnn_steps
        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.cluster_embed_size = cluster_embed_size

    def compute_loss(self):
 
        cluster_mu_sample = tf.placeholder(tf.float32,[None, self.latent_size*self.cluster_embed_size])
        cluster_log_sigm_sq_sample = tf.placeholder(tf.float32,[None, self.latent_size*self.cluster_embed_size])
 
        cluster_mask = tf.placeholder(tf.float32, [None, self.cluster_embed_size*self.latent_size])

        image = tf.placeholder(tf.float32, [None, self.image_embed_size])
        attribute = tf.placeholder(tf.float32, [None, self.cluster_embed_size])
        sentence = tf.placeholder(tf.int32, [None, self.num_rnn_steps + 1])
        mask = tf.placeholder(tf.float32, [None, self.num_rnn_steps + 1])
        mask2 = tf.placeholder(tf.float32, [None, self.num_rnn_steps + 1])
        
        z, mu, log_sigma_sq = self.encoder.encode(image, sentence, mask2, attribute)

        mu_sample = 0
        mu_gt = 0
        exp_sigma_sq_sample = 0
        exp_sigma_sq_gt = 0
        z_cluster_sample = 0

        for id_cluster in range(self.cluster_embed_size):
            mu_sample += tf.slice(cluster_mask, [0,id_cluster*self.latent_size], [-1, self.latent_size])*\
            tf.slice(mu, [0,id_cluster*self.latent_size], [-1, self.latent_size])

            mu_gt += tf.slice(cluster_mask, [0,id_cluster*self.latent_size], [-1, self.latent_size])*\
            tf.slice(cluster_mu_sample, [0,id_cluster*self.latent_size], [-1, self.latent_size])

            exp_sigma_sq_sample += tf.square(tf.slice(cluster_mask, [0,id_cluster*self.latent_size], [-1, self.latent_size]))*\
            tf.slice(tf.exp(log_sigma_sq), [0,id_cluster*self.latent_size], [-1, self.latent_size])

            exp_sigma_sq_gt += tf.square(tf.slice(cluster_mask, [0,id_cluster*self.latent_size], [-1, self.latent_size]))*\
            tf.slice(tf.exp(cluster_log_sigm_sq_sample), [0,id_cluster*self.latent_size], [-1, self.latent_size])

            z_cluster_sample += tf.slice(cluster_mask, [0,id_cluster*self.latent_size], [-1, self.latent_size])*\
            tf.slice(z, [0,id_cluster*self.latent_size], [-1, self.latent_size])


        kl_divergence_allcluster = 1 + tf.log(exp_sigma_sq_sample+0.0001) -  tf.log(exp_sigma_sq_gt+0.0001)\
                - (tf.square(mu_sample - mu_gt) + exp_sigma_sq_sample)/(exp_sigma_sq_gt+0.0000001)

        kl_divergence = - 0.5 * tf.reduce_sum(kl_divergence_allcluster, 1)
        
        reconstruction = self.decoder.decode(image, z_cluster_sample, sentence, mask, attribute)

        reconstruction_loss = tf.reduce_mean(reconstruction)
        kl_divergence_loss = tf.reduce_mean(kl_divergence)

        loss = reconstruction_loss + kl_divergence_loss/10

        return loss, image, sentence, attribute, mask, mask2, reconstruction_loss, kl_divergence_loss, cluster_mu_sample, \
        cluster_log_sigm_sq_sample, cluster_mask


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def truncate_list(l, num):
    if num == -1:
      num = len(l)
    return l[:min(len(l), num)]


def train(dataset, image_embed_size, word_embed_size, rnn_hidden_size, latent_size, initial_learning_rate, momentum,
          num_epochs, num_epochs_to_halve, batch_size, cluster_embed_size, cluster_mu_center, cluster_log_sigm_sq, 
          model_directory, restore_ckpt):
    
    #To avoid leaving dead nodes in the session
    tf.reset_default_graph()

    tf.set_random_seed(456)

    learning_rate = tf.placeholder(tf.float32, shape=[])

    # toy example contains one image and 5 sentences. 
    path_h5 = 'mRNN_coco_train_data.hdf5'
    path_js = 'mRNN_coco_train_data.json'
    
    # 80-D vector indicates which cluster it belongs to
    path_cluster_vec = 'object_label_80.npy'

    h5 = h5py.File(path_h5, 'r')
    js = json.load(open(path_js, 'r'))

    # visual features stored in hdf5 file for speed.
    path_visual_fea = 'img_coco_VggmRNN_trainFea.hdf5'
    img_coco_VggmRNN_trainFea = h5py.File(path_visual_fea, 'r')

    cluster_vec_full = np.load(path_cluster_vec)
    
    num_samples = len(js['images'])
    print num_samples

    max_sentence_length = 16
    print max_sentence_length

    vae = VariationalAutoencoder(image_embed_size, word_embed_size, rnn_hidden_size, max_sentence_length + 1,
                                 len(js['ix_to_word']) + 1, latent_size, cluster_embed_size)

    loss, image, sentence, attribute, mask, mask2, reconstruction_loss, kl_divergence_loss, cluster_mu_sample, \
    cluster_log_sigm_sq_sample, cluster_mask = vae.compute_loss()
    
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, max_to_keep=300)
    
    if restore_ckpt:
        print 'restoring checkpoint'
        saver.restore(sess, restore_ckpt)
        print 'done'

    for epoch in range(0, num_epochs):
        print 'Epoch {}'.format(epoch)

        # save the model every 100 epochs. ---- into its version 1 saver mode. 
        if epoch % 1 == 0:
            save_path = saver.save(sess, "{}/epoch{:03d}.ckpt".format(model_directory, epoch))
            
            print("Model saved in file: {}".format(save_path))

        for batch in range(num_samples / batch_size):

            global_index = batch_size * batch # global indexes over images. 0: image_size -1
            
            num_sentences_batch = h5['label_end_ix'][global_index + batch_size - 1] - h5['label_start_ix'][global_index]
            
            image_batch = np.empty([num_sentences_batch, image_embed_size])
            sentence_batch = np.empty([num_sentences_batch, max_sentence_length])
            cluster_vec_batch = np.empty([num_sentences_batch, cluster_embed_size])
            attribute_batch = np.empty([num_sentences_batch, cluster_embed_size])
            cluster_mask_batch = np.empty([num_sentences_batch, cluster_embed_size*latent_size])

            counter = 0

            for i in range(batch_size):
                num_captions = h5['label_end_ix'][global_index + i] - h5['label_start_ix'][global_index + i]
                # the interface to get the image features, which is extracted using vgg16_COCO_LW.py here. 
                image_batch[counter: counter + num_captions, :] = \
                    img_coco_VggmRNN_trainFea['img_coco_VggmRNN_trainFea'][global_index + i]

                sentence_batch[counter: counter + num_captions, :] = \
                    h5['labels'][h5['label_start_ix'][global_index + i]:h5['label_end_ix'][global_index + i], :]

                cluster_vec_batch[counter: counter + num_captions,:] = cluster_vec_full[global_index+i,:]
                attribute_batch[counter: counter + num_captions,:] = cluster_vec_full[global_index+i,:]
                
                counter += num_captions

            sentence_batch = np.concatenate(
                (np.zeros((image_batch.shape[0], 1)), sentence_batch, np.zeros((image_batch.shape[0], 1))), axis=1)

            mask_batch = (sentence_batch != 0)
            mask_batch[:, 0] = True
            mask2_batch = np.zeros_like(mask_batch, dtype=float)
            for j in range(mask_batch.shape[0]):
                row_sum = sum(mask_batch[j, :])
                mask2_batch[j, row_sum - 1] = 1
            mask_batch = mask_batch.astype(float)

            cluster_vec_weight = np.transpose(1/(np.tile(np.sum(cluster_vec_batch, 1)+0.0000001, [cluster_embed_size,1])))

            cluster_mask_batch = np.kron(np.multiply(cluster_vec_batch,cluster_vec_weight),np.ones(latent_size))
            #pdb.set_trace()

            cluster_mu_sample_batch = np.tile(cluster_mu_center, [num_sentences_batch,1])
            cluster_log_sigm_sq_sample_batch = np.tile(cluster_log_sigm_sq, [num_sentences_batch,1])

            lr = initial_learning_rate / (2 ** (epoch / num_epochs_to_halve))

            train_loss, train_reconstruction_loss, train_kl_divergence_loss, _ = sess.run(
                [loss, reconstruction_loss, kl_divergence_loss, optimizer],
                feed_dict={learning_rate: lr, \
                           image: image_batch, \
                           sentence: sentence_batch, \
                           attribute: attribute_batch, \
                           mask: mask_batch,
                           mask2: mask2_batch, \
                           cluster_mu_sample: cluster_mu_sample_batch,\
                           cluster_log_sigm_sq_sample: cluster_log_sigm_sq_sample_batch,\
                           cluster_mask: cluster_mask_batch})

            print 'iter_num:{}'.format(batch)
            print 'epoch_num: {}, lr: {}, loss: {}, reconstr: {}, kl_div: {}'.format(epoch, lr, train_loss,
                                                                      train_reconstruction_loss,
                                                                      train_kl_divergence_loss)

    sess.close()


def compute_max_bleu(dataset, image_embed_size, word_embed_size, rnn_hidden_size, latent_size, initial_learning_rate,
                 momentum, num_epochs, num_epochs_to_halve, batch_size, cluster_embed_size, cluster_mu_center, 
                 cluster_log_sigm_sq, model_directory, test_epoch,
                 beam_size=0, num_samples_per_val=500):

    tf.reset_default_graph()

    print 'beam search generating now !'

    h5_val = h5py.File('mRNN_coco_val_data.hdf5', 'r')
    js_val = json.load(open('mRNN_coco_val_data.json', 'r'))
    
    h5_train = h5py.File('mRNN_coco_train_data.hdf5', 'r')
    js_train = json.load(open('mRNN_coco_train_data.json', 'r'))
    
    ### load the predicted object categories. 
    path_val_attr = 'coco_detector/coco_scores_binary.npy'
    attribute_val = np.load(path_val_attr)
    obj_class = json.load(open('coco_detector/coco_classes.json','r'))
    obj_class = np.array(obj_class)
    #pdb.set_trace()
    class_name = np.load('class_name.npy')
    ### to match the class_name with the obj_class. 
    list_sort_idx = np.argsort(class_name)
    idx = np.argsort(list_sort_idx)
    attribute_val = attribute_val[:,idx]
    ## attribute_val is the index matrix telling which cluster the image belongs to: 
    
    num_val_samples = len(js_val['images'])
    js_train['ix_to_word']['0'] = '<END>'

    max_sentence_length = h5_train['labels'].shape[1]

    vae = VariationalAutoencoder(image_embed_size, word_embed_size, rnn_hidden_size, max_sentence_length + 1,
                                 len(js_train['ix_to_word']), latent_size, cluster_embed_size)

    vae.compute_loss()
    
    model = '{}/ver2_epoch{}.ckpt'.format(model_directory, test_epoch)

    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()

    image, attribute, latent, logit_init, state_init = vae.decoder.decode_beam_initial()
    word_idx, state, state_current, logits = vae.decoder.decode_beam_continue()
  
    cluster_mu_matrix = np.reshape(cluster_mu_center,[cluster_embed_size,latent_size])

    Predict_sentence_results_txt_File = open(model_directory+'/Predict_sentence_results_txt_File.txt','w')

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model)
    print("Model {} restored.".format(model))

    list_of_reference_questions = list()
    list_of_generated_captions = list()
    list_of_object_labels = list()
    list_of_cluster_flags = list()
    list_of_candidate_questions = list()
    list_of_bleu_scores = list()
    list_of_mRNN_bleu_scores = list()
    list_of_mRNN_reRank = list()
    list_of_clusterVAE_captions = list()
    list_of_imgID = np.zeros(1000)
    list_of_candidate_beamsearc_questions = list()
    list_of_VAE_captions = list()

    mRNN_sens = json.load(open('list_of_mRNN_sentences.json', 'r'))

    # out file will store the generated results that will be evaluated in MSCOCO official BLEU evaluation script. 
    out = []

    bleus_1 = list()
    bleus_2 = list()
    bleus_3 = list()
    bleus_4 = list()
    
    number_caption = 0

    for id_image in range(1000):
        start = timeit.default_timer()

        jimg = {}

        print id_image

        local_num_questions = h5_val['label_end_ix'][id_image] - h5_val['label_start_ix'][id_image]

        # cluster_label is the predicted binary labels telling which clusters the image should belong to:
        cluster_label = attribute_val[id_image,:]
        cluster_label_idx = np.where(cluster_label>0)
        cluster_label_idx = cluster_label_idx[0]

        ### if the label vector is empty, just go over all categories.
        if len(cluster_label_idx)==0:
            cluster_label_idx = np.arange(80)

        reference_questions = list()

        for j in range(local_num_questions):
            label = h5_val['labels'][h5_val['label_start_ix'][id_image] + j]
            reference_question = list()
            for x in label:
                if x != 0:
                    reference_question.append(js_val['ix_to_word'][x.astype('str')])
                else:
                    break
            reference_questions.append(reference_question)
        list_of_reference_questions.append(reference_questions)

        ### the new mean of the distribution generating z is the linear combination of cluster means.
        
        if len(cluster_label_idx) > 0:
            obj_pred = class_name[cluster_label_idx]
        else:
            obj_pred = 'None'

        list_of_object_labels.append(obj_pred)
        
        list_of_captions_cluster = list()
        ### flag_cluster memorize which cluster the generated sentence come from. 
        flag_cluster = []

        z_mean = 0

        for idx_center in range(len(cluster_label_idx)):
            id_cluster = cluster_label_idx[idx_center]
            z_mean = z_mean + cluster_mu_matrix[id_cluster]

        z_mean = z_mean/len(cluster_label_idx)

        z_test_size = 20

        list_of_candidate_questions_perImage = list()

        for idx_batch in range(z_test_size):

            z_batch = []

            z_batch.append(np.random.multivariate_normal(z_mean, 2*np.diag(np.ones(latent_size)), 1))

            z_batch = np.squeeze(np.array(z_batch))

            img_fea = np.load('coco_vgg16_fc7_features/{}.npy'.format(js_val['images'][id_image]['file_path']))

            list_of_imgID[id_image] = js_val['images'][id_image]['id']

            image_batch_test = np.tile(img_fea,(1,1))
            attribute_batch = np.tile(cluster_label,(1,1))
            z_batch = np.tile(z_batch, (1, 1))

            good_sentences = [] # store sentences already ended with <bos>
            cur_best_cand = [] # store current best candidates
            highest_score = 0.0 # hightest log-likelihodd in good sentences
            
            #pdb.set_trace()
            logit_init_batch, state_init_batch = sess.run([logit_init, state_init], \
                feed_dict={image: image_batch_test, attribute: attribute_batch, latent: z_batch})

            logit_init_batch = np.squeeze(logit_init_batch)
            logit_init_batch = softmax(logit_init_batch)
            logit_init_order = np.argsort(-logit_init_batch)
            #pdb.set_trace()

            for ind_b in range(beam_size):
              cand = {}
              cand['indexes'] = [logit_init_order[ind_b]]
              cand['score'] = -np.log(logit_init_batch[logit_init_order[ind_b]])
              cand['state'] = state_init_batch
              cur_best_cand.append(cand)


            # Expand the current best candidates until max_steps or no candidate
            for i in range(max_sentence_length):
              # move candidates end with <END> to good_sentences or remove it
              cand_left = []
              for cand in cur_best_cand:
                if len(good_sentences) > beam_size and cand['score'] > highest_score:
                  continue # No need to expand that candidate
                if cand['indexes'][-1] == 0:
                  good_sentences.append(cand)
                  highest_score = max(highest_score, cand['score'])
                else:
                  cand_left.append(cand)
              cur_best_cand = cand_left
              if not cur_best_cand:
                break
              # expand candidate left
              cand_pool = []
              #pdb.set_trace()
              #word_idx, state, state_current, logits = vae.decoder.decode_beam_continue()
              for cand in cur_best_cand:
                # Get the continued states. 
                
                #pdb.set_trace()
                state_current_batch, logits_batch = sess.run([state_current, logits], feed_dict={state:cand['state'], \
                    word_idx: np.reshape(cand['indexes'][-1],[-1])})
                #pdb.set_trace()
                logits_batch = np.squeeze(logits_batch)
                logits_batch = softmax(logits_batch)
                logit_order = np.argsort(-logits_batch)
                for ind_b in xrange(beam_size):
                  cand_e = copy.deepcopy(cand)
                  cand_e['indexes'].append(logit_order[ind_b])
                  #pdb.set_trace()
                  cand_e['score'] -= np.log(logits_batch[logit_order[ind_b]])
                  cand_e['state'] = state_current_batch
                  cand_pool.append(cand_e)
              # get final cand_pool
              cur_best_cand = sorted(cand_pool, key=lambda cand: cand['score'])
              #pdb.set_trace()
              cur_best_cand = truncate_list(cur_best_cand, beam_size)

                # Add candidate left in cur_best_cand to good sentences
            for cand in cur_best_cand:
                if len(good_sentences) > beam_size and cand['score'] > highest_score:
                    continue
                if cand['indexes'][-1] != 0:
                    cand['indexes'].append(0)
                good_sentences.append(cand)
                highest_score = max(highest_score, cand['score'])
            
            # Sort good sentences and return the final list
            good_sentences = sorted(good_sentences, key=lambda cand: cand['score'])
            #pdb.set_trace()
            good_sentences = truncate_list(good_sentences, beam_size)
            #pdb.set_trace()
            Q = list()
            for j in range(len(good_sentences)):
                question_new = ''
                question_batch = good_sentences[j]['indexes']
                for x in question_batch:
                    if x != 0:
                        question_new += js_train['ix_to_word'][x.astype('str')] + ' '
                    else:
                        break
                Q.append(question_new[:-1])

            candidate_questions = list()
            for q in Q:
                q_split = q.split()
                candidate_questions.append(q_split)

            
            list_of_candidate_questions_perImage.append(candidate_questions[0])

        Q = list()
        for id_sen in range(len(list_of_candidate_questions_perImage)):
            question_new = ''
            for x in list_of_candidate_questions_perImage[id_sen]:
                if x != 0:
                    question_new += x + ' '
                else:
                    break
            Q.append(question_new[:-1])

        #pdb.set_trace()
        unique_questions = []
        unique_questions = list(set(Q))
        unique_questions = [uq for uq in unique_questions if uq]
        #pdb.set_trace()

        number_caption = number_caption + len(unique_questions)

        candidate_unique_questions = list()

        for q in unique_questions:
            q_split = q.split()
            candidate_unique_questions.append(q_split)

        bleus = []
        for id_cand in range(len(candidate_unique_questions)):
            bleus.append(nltk.translate.bleu_score.sentence_bleu(reference_questions, 
                candidate_unique_questions[id_cand], smoothing_function=smoothing_function.method2))

        ## reRank the mRNN sentences by bleu scores. 
        index_b = np.argsort(bleus)

        list_of_VAE_captions.append(candidate_unique_questions)

        generated_caption_Evalcoco = ' '.join(str(p) for p in candidate_unique_questions[index_b[-1]])
        #pdb.set_trace()
        # image id in the val set for current index of i 
        image_id = js_val['images'][id_image]['id']
        jimg['image_id'] = image_id
        jimg['caption'] = generated_caption_Evalcoco
        out.append(jimg)
    
    np.save('data_save_path_{}.npy'.format(test_epoch),list_of_VAE_captions)
    json.dump(out, open('data_save_path_{}.json'.format(test_epoch), 'w'))

    print 'number of averaged unique captions:'
    print number_caption/1000.0
    
    sess.close()


 
if __name__ == '__main__':
    # model hyper-parameters

    image_embed_size = 4096
    word_embed_size = 256

    cluster_embed_size = 80

    rnn_hidden_size = 512
    ############################################################################
    latent_size = 150
    ############################################################################
    # training hyper-parameters  #### 0.01
    initial_learning_rate = 0.01
    momentum = 0.90  # default is 0.9. 
    num_epochs = 1000

    # to halve the learing rate after each 5 epochs.
    num_epochs_to_halve = 5
    batch_size = 100  ######### default is 100 for all. 

    dataset = 'coco'
    
    #beam_size = 0
    num_samples_per_val = 100

    generate_cluster = False  

    tf.set_random_seed(456)
    np.random.seed(123)

    model_directory = 'tf_v1.4_AG-CVAE_witOBJ_clusterDim_{}_latentdim_{}_std_0.1_lr_{}_combinedStd'.format(cluster_embed_size, latent_size, initial_learning_rate)

    restore_fname = None
    if len(sys.argv) > 1:
        restore_fname = sys.argv[1]
    
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    if generate_cluster:
        
        cluster_mu_matrix = list()
        for id_cluster in range(cluster_embed_size):

            cluster_item = 2*np.random.random_sample((latent_size,)) - 1
            cluster_item = cluster_item/(np.sqrt(np.sum(cluster_item**2)))
            cluster_mu_matrix.append(cluster_item)
           
        np.save(model_directory + '/cluster_mu_COCO.npy', cluster_mu_matrix)
        cluster_mu_matrix = np.squeeze(cluster_mu_matrix)
        print cluster_mu_matrix.shape
        
    else:
        cluster_mu_matrix = np.load('tf_v1.4_AG-CVAE_witOBJ_clusterDim_80_latentdim_150_std_0.1_lr_0.01_combinedStd/cluster_mu_COCO.npy')
        cluster_mu_matrix = np.squeeze(cluster_mu_matrix)

    cluster_mu_center = np.reshape(cluster_mu_matrix, [1, latent_size*cluster_embed_size])

    cluster_log_sigm_sq_vec = np.log(np.square(0.1))*np.ones(latent_size)  # -4.60 is ln(0.01) std=0.1 for each cluster center. 
    cluster_log_sigm_sq =np.tile(cluster_log_sigm_sq_vec,[1, cluster_embed_size])

    ################################################################################################################

    train(dataset, 
        image_embed_size, 
        word_embed_size, 
        rnn_hidden_size, 
        latent_size, 
        initial_learning_rate, 
        momentum,
        num_epochs, 
        num_epochs_to_halve, 
        batch_size,
        cluster_embed_size,
        cluster_mu_center,
        cluster_log_sigm_sq, 
        model_directory,
        restore_fname)

    '''
    beam_size = 2
    test_epoch = '015'
    compute_max_bleu(dataset, 
        image_embed_size, 
        word_embed_size, 
        rnn_hidden_size, 
        latent_size, 
        initial_learning_rate,
        momentum, 
        num_epochs, 
        num_epochs_to_halve, 
        batch_size, 
        cluster_embed_size, 
        cluster_mu_center, 
        cluster_log_sigm_sq,
        model_directory, 
        test_epoch,
        beam_size)
    '''