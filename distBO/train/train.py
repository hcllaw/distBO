from __future__ import division, print_function
from functools import partial

import tensorflow as tf
import numpy as np
from sklearn.utils import check_random_state

from gp_utils import loop_batches
from distBO.utils import check_dims

def train_network(sess, net, train, initialise=True, model_type=None, gradient_clip=False,
                  max_epochs=100, first_early_stop_epoch=None, batch_size=250, stratify=True,
                  optimizer=tf.train.AdamOptimizer, lr=0.005, seed=23,
                  display_every=5):
    
    if first_early_stop_epoch is None:
        first_early_stop_epoch = max_epochs // 5

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    cur_min = np.inf  # used for early stopping
    countdown = np.inf

    rs = check_random_state(seed)
    looper = partial(loop_batches, batch_size=batch_size,
                     max_epochs=max_epochs, rs=rs, model_type=model_type,
                     stratify=stratify)

    if initialise:
        with tf.control_dependencies(update_ops):
            if gradient_clip:
                optimizer = optimizer(lr)
                gradients, variables = zip(*optimizer.compute_gradients(net.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0) # can also use 5.0
                net.optimize_step = optimizer.apply_gradients(zip(gradients, variables))
                print('Gradient Clip')
            else:
                optimizer = optimizer(lr)
                gradients, variables = zip(*optimizer.compute_gradients(net.loss))
                net.optimize_step = optimizer.minimize(net.loss)
        print('Intialise parameters')
        sess.run(tf.global_variables_initializer())
    for epoch, (batch_X, batch_y, embed_sizes) in enumerate(looper(train)):
        train['batch_X'] = batch_X
        train['batch_y'] = batch_y
        train['embed_sizes'] = embed_sizes
        if epoch == 0:
            update_sum_matrix = True
        else:
            update_sum_matrix = False
        _, loss = sess.run(
                  [net.optimize_step, net.loss], feed_dict=net.feed_dict(train, update_sum_matrix=update_sum_matrix))
        if epoch >= first_early_stop_epoch:
            if (cur_min - loss) > 0.0001 and epoch >= first_early_stop_epoch:
                countdown = 100
                cur_min = loss
            else:
                countdown -= 1
        if epoch % display_every == 0:
            s = ("{: 4d}: train loss = {:8.5f}"
                 ).format(epoch, loss)
            print(s)

        if epoch >= first_early_stop_epoch and countdown <= 0:
            break

    if epoch >= first_early_stop_epoch:
        print(("Stopping at epoch {} with loss {:.8}\n".format(epoch, loss)))
    else:
        print("Using final model with loss")
    return loss

def eval_network(sess, net, train, test, update_sum_matrix=False, use_embed=False):
    d = net.feed_dict(train, data_pred=test, update_sum_matrix=update_sum_matrix, use_embed=use_embed)
    mean, var = sess.run([net.pred_mean, net.pred_var], feed_dict=d)
    return mean, var

def embed_network(sess, net, train, test, algorithm='GP', update_sum_matrix=False):
    d = net.feed_dict(train, data_pred=test, update_sum_matrix=update_sum_matrix)
    if algorithm == 'GP':
        dist, dist_pred = sess.run([net.k_dist, net.k_dist_pred], feed_dict=d)
    elif algorithm == 'BLR':
        dist, dist_pred = sess.run([net.dist_fea, net.dist_fea_pred], feed_dict=d)
    return dist, dist_pred

def sim_network(sess, net, train, test, update_sum_matrix=False):
    d = net.feed_dict(train, data_pred=test, sim=True, 
                      update_sum_matrix=update_sum_matrix)
    if net.dist:
        similarities = sess.run([net.k_dist_pred], feed_dict=d)
    elif net.multi:
        similarities = sess.run([net.k_task_pred], feed_dict=d)
    elif net.manual:
        similarities = sess.run([net.k_man_pred], feed_dict=d)
    return similarities