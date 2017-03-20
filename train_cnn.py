import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import tensorflow as tf
from text_cnn import TextCNN
from sklearn.model_selection import train_test_split
import datetime

logging.getLogger().setLevel(logging.INFO)


def train_cnn():
    input_file = sys.argv[1]
    if os.path.exists('./data/x.p') and \
            os.path.exists('./data/y.p') and \
            os.path.exists('./data/vocabulary.p') and \
            os.path.exists('./data/vocabulary_inv.p') and \
            os.path.exists('./data/labels.p'):
        x_ = pickle.load(open("./data/x.p", "rb"))
        y_ = pickle.load(open("./data/y.p", "rb"))
        vocabulary = pickle.load(open("./data/vocabulary.p", "rb"))
        vocabulary_inv = pickle.load(open("./data/vocabulary_inv.p", "rb"))
        labels = pickle.load(open("./data/labels.p", "rb"))
    else:
        x_, y_, vocabulary, vocabulary_inv, _, labels = data_helper.load_data(input_file)

    training_config = sys.argv[2]
    params = json.loads(open(training_config).read())

    # Assign a n dimension vector to each word
    word_embeddings = data_helper.load_embeddings(vocabulary, dim=params['embedding_dim'])
    embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # Split the original dataset into train set and test set
    x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)

    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    timestamp = str(int(time.time()))
    trained_dir = './trained_results_' + timestamp + '/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                embedding_mat=embedding_mat,
                non_static=params['non_static'],
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            # optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False, name='Adam')
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint files will be saved in this directory during training
            checkpoint_dir = './checkpoints_' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob'],
                }
                _, step, summaries, loss_, accuracy_ = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_, accuracy_))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                }
                step, summaries, loss_, accuracy_, predictions_ = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("evaluation on test set:")
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_, accuracy_))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy_, loss_, predictions_

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])

            # Train the model with x_train and y_train
            i = 0
            for train_batch in train_batches:
                logging.info('Training on batch: {}'.format(i))
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                # Evaluate the model with x_dev and y_dev
                if current_step % params['evaluate_every'] == 0:
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)

                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        acc, loss, predictions = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)

                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.critical('Saved model {} at step {}'.format(path, i))
                i += 1
            logging.critical('Training is complete, testing the best model on x_test and y_test')

            # Evaluate x_test and y_test
            saver.restore(sess, checkpoint_prefix + '-' + str(i))
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
                total_test_correct += int(num_test_correct)
            logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

    # Save trained parameters and files since predict.py needs them
    with open(trained_dir + 'words_index.json', 'w') as outfile:
        json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
        pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    with open(trained_dir + 'labels.json', 'w') as outfile:
        json.dump(labels, outfile, indent=4, ensure_ascii=False)

    os.rename(path, trained_dir + 'best_model.ckpt')
    os.rename(path + '.meta', trained_dir + 'best_model.meta')
    shutil.rmtree(checkpoint_dir)
    logging.critical('{} has been removed'.format(checkpoint_dir))

    params['sequence_length'] = x_train.shape[1]
    with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
        json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
    train_cnn()
