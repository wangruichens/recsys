# Ref: https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
# Multi-DAE and Multi-VAE(PR partially-regularized)
import os
import shutil
import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer
import bottleneck as bn
from vae_cf_model import MultiVAE, MultiDAE

pro_dir = '.'
unique_sid = list()
with open('unique_sid.txt', 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)


def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                              (rows, cols)), dtype = 'float64',
                             shape = (n_users, n_items))
    print('train data shape :', data.shape)
    return data


# about 11w users and 2w items
train_data = load_train_data('train.csv')
print(train_data.shape)
print(train_data[0])
print('load train data finished..')


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                 (rows_tr, cols_tr)), dtype = 'float64', shape = (end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                 (rows_te, cols_te)), dtype = 'float64', shape = (end_idx - start_idx + 1, n_items))
    print('tr shape :', data_tr.shape)
    print('te shape :', data_te.shape)
    return data_tr, data_te


vad_data_tr, vad_data_te = load_tr_te_data('validation_tr.csv', 'validation_te.csv')

N = train_data.shape[0]
# Users idx
idxlist = range(N)

# training batch size
batch_size = 500
batches_per_epoch = int(np.ceil(float(N) / batch_size))

N_vad = vad_data_tr.shape[0]
idxlist_vad = range(N_vad)

# validation batch size (since the entire validation set might not fit into GPU memory)
batch_size_vad = 2000

# the total number of gradient updates for annealing
total_anneal_steps = 200000
# largest annealing parameter
anneal_cap = 0.2


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis = 1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis = 1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis = 1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis = 1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis = 1)
    X_pred_binary = np.zeros_like(X_pred, dtype = bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis = 1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis = 1))
    return recall


p_dims = [200, 600, n_items]

tf.reset_default_graph()
vae = MultiVAE(p_dims, lam = 0.0, random_seed = 98765)

# logits(predict value), loss (ELBO+KL)
saver, logits_var, loss_var, train_op_var, merged_var = vae.build_graph()

ndcg_var = tf.Variable(0.0)
ndcg_dist_var = tf.placeholder(dtype = tf.float64, shape = None)
ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])

arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))

log_dir = './log/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps / 1000, anneal_cap, arch_str)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph = tf.get_default_graph())

ckpt_dir = './ckpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
    total_anneal_steps / 1000, anneal_cap, arch_str)

if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

print("ckpt directory: %s" % ckpt_dir)

n_epochs = 200

ndcgs_vad = []
# Do train or Do test
do_train = True
do_eval = True

if do_train:
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        update_count = 0.0

        for epoch in range(n_epochs):
            np.random.shuffle(list(idxlist))
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]

                # X stands for user interactions with items
                if sparse.isspmatrix(X):
                    X = X.toarray()
                # X.shape = (batch size, item size)
                X = X.astype('float32')
                # print(len(X[0]))

                if total_anneal_steps > 0:
                    anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                else:
                    anneal = anneal_cap

                feed_dict = {vae.input_ph: X,
                             vae.keep_prob_ph: 0.5,
                             vae.anneal_ph: anneal,
                             vae.is_training_ph: 1}
                sess.run(train_op_var, feed_dict = feed_dict)

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_train,
                                               global_step = epoch * batches_per_epoch + bnum)

                update_count += 1

            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict = {vae.input_ph: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict = {ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(ckpt_dir))
                best_ndcg = ndcg_
    # plt.figure(figsize = (12, 3))
    # plt.plot(ndcgs_vad)
    # plt.ylabel("Validation NDCG@100")
    # plt.xlabel("Epochs")
    # plt.show()
if do_train or do_eval:
    test_data_tr, test_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'),
        os.path.join(pro_dir, 'test_te.csv'))
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    batch_size_test = 2000
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam = 0.0)
    saver, logits_var, _, _, _ = vae.build_graph()
    ckpt_dir = './ckpt/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        total_anneal_steps / 1000, anneal_cap, arch_str)
    print("ckpt directory: %s" % ckpt_dir)
    n100_list, r20_list, r50_list = [], [], []

    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(ckpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            pred_val = sess.run(logits_var, feed_dict = {vae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 100))
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 20))
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

train_dae = False
if train_dae and do_train:
    p_dims = [200, n_items]
    tf.reset_default_graph()
    dae = MultiDAE(p_dims, lam = 0.01 / batch_size, random_seed = 98765)

    saver, logits_var, loss_var, train_op_var, merged_var = dae.build_graph()

    ndcg_var = tf.Variable(0.0)
    ndcg_dist_var = tf.placeholder(dtype = tf.float64, shape = None)
    ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
    ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
    merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in dae.dims[1:-1]]))
    log_dir = './log/DAE/{}'.format(arch_str)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    print("log directory: %s" % log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph = tf.get_default_graph())
    ckpt_dir = './ckpt/DAE/{}'.format(arch_str)

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    print("ckpt directory: %s" % ckpt_dir)

    n_epochs = 200
    ndcgs_vad = []

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        best_ndcg = -np.inf

        for epoch in range(n_epochs):
            np.random.shuffle(list(idxlist))
            # train for one epoch
            for bnum, st_idx in enumerate(range(0, N, batch_size)):
                end_idx = min(st_idx + batch_size, N)
                X = train_data[idxlist[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                feed_dict = {dae.input_ph: X,
                             dae.keep_prob_ph: 0.5}
                sess.run(train_op_var, feed_dict = feed_dict)

                if bnum % 100 == 0:
                    summary_train = sess.run(merged_var, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_train, global_step = epoch * batches_per_epoch + bnum)

            # compute validation NDCG
            ndcg_dist = []
            for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                end_idx = min(st_idx + batch_size_vad, N_vad)
                X = vad_data_tr[idxlist_vad[st_idx:end_idx]]

                if sparse.isspmatrix(X):
                    X = X.toarray()
                X = X.astype('float32')

                pred_val = sess.run(logits_var, feed_dict = {dae.input_ph: X})
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                ndcg_dist.append(NDCG_binary_at_k_batch(pred_val, vad_data_te[idxlist_vad[st_idx:end_idx]]))

            ndcg_dist = np.concatenate(ndcg_dist)
            ndcg_ = ndcg_dist.mean()
            ndcgs_vad.append(ndcg_)
            merged_valid_val = sess.run(merged_valid, feed_dict = {ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
            summary_writer.add_summary(merged_valid_val, epoch)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                saver.save(sess, '{}/model'.format(ckpt_dir))
                best_ndcg = ndcg_
            # plt.figure(figsize=(12, 3))
            # plt.plot(ndcgs_vad)
            # plt.ylabel("Validation NDCG@100")
            # plt.xlabel("Epochs")
            # plt.show()
if train_dae and (do_train or do_eval):

    n100_list, r20_list, r50_list = [], [], []

    with tf.Session() as sess:
        saver.restore(sess, '{}/model'.format(ckpt_dir))

        for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            end_idx = min(st_idx + batch_size_test, N_test)
            X = test_data_tr[idxlist_test[st_idx:end_idx]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = X.astype('float32')

            pred_val = sess.run(logits_var, feed_dict = {dae.input_ph: X})
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 100))
            r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 20))
            r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k = 50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
