import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dataGenerator import dataGenerator
from model import *
import tensorflow as tf
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
nsml_avail = True
try:
    import visdom
    from nsml import Visdom
    viz = Visdom(visdom=visdom)
except:
    nsml_avail = False

import pdb
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def errorfill(x, y, yerr, label, color=None, marker = 'o', alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    base_line, = ax.plot(x, y, color=color,marker = 'o',label = label)
    if color is None:
        color = base_line.get_color()
    ax.fill_between(x, ymax, ymin, facecolor=color, alpha=alpha_fill)

class Logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir,exist_ok=True)
        # self.loss = get_loss_function(loss_name, pos_weight)
        self.writer = tf.summary.FileWriter(log_dir)
        self.loss_train = []
        self.loss_test = []
        self.overlap_train = []
        self.overlap_test = []
        self.args = {}

    def add_train_loss(self, loss, iter_count):
        self.loss_train.append(loss.item())
        self.scalar_summary('loss', loss.item(), iter_count+1)

    # def add_test_loss(self, loss):
    #     self.loss_test.append(loss.data.cpu().numpy())

    def add_train_overlap(self, pred, labels, iter_count):
        overlap = compute_overlap(pred, labels)
        self.overlap_train.append(overlap)
        self.scalar_summary('overlap', overlap, iter_count+1)

    # def add_test_overlap(self, pred, labels):
    #     self.overlap_test.append(compute_overlap(pred, labels))

    def write_settings(self, args):
        with open(args.save_path+"experiment_settings.txt", "w") as f:
            for arg in vars(args):
                param = str(arg)
                value = getattr(args, arg)
                f.write('Param: {:<20} Value: {:<10}\n'.format(param, value))
                self.args[param] = value



    def save_model(self, model,optim,epoch,batch_size):
        self.epoch = epoch
        self.batch_size = batch_size
        try:
            os.stat(self.args['save_path'])
        except:
            os.mkdir(self.args['save_path'])
        save_dir = os.path.join(self.args['save_path'], 'models/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)

        filename = 'gnn.pt'
        path = os.path.join(save_dir, filename)

        data = {'model':model.state_dict(),
                'optimizer':optim.state_dict(),
                'epoch':epoch+1,
                'loss':self.loss_train,
                'overlap':self.overlap_train
                }
        torch.save(data, path)
        # if batch_size > 1:
        #     save_freq = 10000
        # elif batch_size == 1:
        #     save_freq = 40000
        # if (epoch>0 and epoch%save_freq==0):
        torch.save(data, save_dir+'gnn_epoch_{}.pt'.format(epoch))
        print('Model Saved.')


    def save_results(self):
        save_dir = os.path.join(self.args['save_path'], 'results/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        filename = 'results'
        path = os.path.join(save_dir, filename)
        np.savez(path,
                 loss_train=np.array(self.loss_train),
                 # loss_test=np.array(self.loss_test),
                 overlap_train = np.array(self.overlap_train)
                 # overlap_test = np.array(self.overlap_test)
                 )
        # if self.batch_size > 1:
        #     save_freq = 5000
        # elif self.batch_size == 1:
        #     save_freq = 20000
        # if (self.epoch>0 and self.epoch%self.batch_size==0):
        #     np.savez(save_dir+'results_epoch_{}'.format(self.epoch),
        #              loss_train=np.array(self.loss_train),
        #              loss_test=np.array(self.loss_test),
        #              overlap_train = np.array(self.overlap_train),
        #              overlap_test = np.array(self.overlap_test)
        #              )
        print('Results Saved.')

    # def load_model(self):
    #     load_dir = os.path.join(self.args['load_path'], 'models/')
    #     # check if any training has been done before.
    #     try:
    #         os.stat(load_dir)
    #     except:
    #         print("Training has to be done before testing. This session will be terminated.")
    #         sys.exit()
    #     path = os.path.join(load_dir, 'gnn.pt')
    #     print('Loading the most recent model...')
    #
    #     return torch.load(path)
    #
    # def load_results(self):
    #     data = np.load(self.args['load_path']+'results/results.npz')
    #     return data

    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        bin_size = 10
        iters = np.linspace(0, len(self.loss_train), num=int(len(self.loss_train)/bin_size))
        y = np.empty(int(len(self.loss_train)/bin_size))
        for i in range(len(y)):
            y[i] = np.sum(self.loss_train[i:i+int(bin_size)])/bin_size
        plt.semilogy(iters, y, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss')
        try:
            os.stat(self.args['save_path']+'plots/')
        except:
            os.mkdir(self.args['save_path']+'plots/')
        path = self.args['save_path'] + 'plots/loss.png'
        plt.ylim([1e-4,1.])
        plt.savefig(path)
        if nsml_avail and len(y)>0:
            viz.line(X=iters, Y=y, win="loss", opts = {'ytype':'log', 'title':'loss'})

    def plot_train_overlap(self):
        plt.figure(1)
        plt.clf()
        bin_size = 10
        iters = np.linspace(0, len(self.overlap_train), num=len(self.overlap_train)/bin_size)
        y = np.empty(int(len(self.overlap_train)/bin_size))
        for i in range(len(y)):
            y[i] = np.mean(self.overlap_train[i:i+bin_size])
        plt.plot(iters, y, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Overlap')
        plt.title('Overlap')
        try:
            os.stat(self.args['save_path']+'plots/')
        except:
            os.mkdir(self.args['save_path']+'plots/')
        path = path = self.args['save_path'] + 'plots/overlap.png'
        plt.ylim([0.,1.])
        plt.savefig(path)
        if nsml_avail and len(y)>0:
            viz.line(X=iters, Y=y, win="overlap", opts = {'title':'overlap'})

        # viz.line(X = iters, Y=y, win="overlap")

    def plot_layer_activation(self, model):
        # plot activation rate for all layers
        try:
            os.stat('./activation_ratio')
        except:
            os.mkdir('./activation_ratio')

        plt.figure(0)
        plt.figure(figsize=(30,15))
        plt.clf()
        for name, module in model.named_modules():
            try:
                act = np.array(module.activation_ratio)
                plt.plot(range(len(act)), act, label=name)
            except:
                pass
        plt.legend()
        plt.title('Activation Rate vs. Time')
        plt.savefig(self.args['save_path']+'activation_ratio/act.png')


    def plot_test_overlap(self, data, x, y, yerr, epoch=None):
        # save and plot
        if epoch is None:
            path_data = self.args['save_path'] + 'test_data/test_data_final.npz'
            path_plot = self.args['save_path'] + 'test_plot/test_overlap_final.png'
        else:
            path_data = self.args['save_path'] + 'test_data/test_data_{}.npz'.format(epoch)
            path_plot = self.args['save_path'] + 'test_plot/test_overlap_{}.png'.format(epoch)
        os.makedirs(self.args['save_path']+'test_data',exist_ok=True)
        os.makedirs(self.args['save_path']+'test_plot',exist_ok=True)
        # try:
        #     os.stat('test_data')
        # except:
        #     os.mkdir('./test_data')
        # try:
        #     os.stat('./test_plot')
        # except:
        #     os.mkdir('./test_plot')


        bp_data = np.load('./data/bp_data_{}.npz'.format(self.args['N']))

        np.savez(path_data, d = data, x = x, y = y, yerr= yerr)
        plt.figure(0)
        plt.clf()
        errorfill(x, y, yerr, color='green', alpha_fill = 0.1, marker='o', label='GNN')
        errorfill(bp_data['x1'], bp_data['y1'], bp_data['yerr1'], color='blue', alpha_fill = 0.1, marker='o', label='BP')
        # plt.errorbar(x, y, yerr, color = 'green', ecolor = 'lightgreen', marker = 'o', label='gnn')
        # plt.errorbar(bp_data['x1'], bp_data['y1'], bp_data['yerr1'], color = 'blue', ecolor = 'lightblue', marker = 'o', label='bp')
        plt.plot(x, (y - bp_data['y1']), color='brown',linewidth=2, label='Diff')
        plt.ylim([-0.2,1.2])
        if self.args['N'] == 1000:
            plt.axvline(x=0.8018625801305059, color='black')  # where bp-1000 hits 0.5
        elif self.args['N'] == 2000:
            plt.axvline(x=0.7490827724624296, color='black')  # where bp-2000 hits 0.5
        plt.axhline(y=0.5, color='black')
        if epoch is not None:
            title = 'Test: Epoch {}. Diff = {:<10.5f}'.format(str(epoch), (y - bp_data['y1']).sum())
            plt.title(title)
        else:
            title = 'Test: Final. Diff = {:<10.5f}'.format((y - bp_data['y1']).sum())
            plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(path_plot)
        # viz.line(X = iters, Y=y, win="loss")

        # # visdom plot
        # viz = Visdom()
        # viz.matplot(plt, win = title)

        # visdom example code below
        # viz.line(X=np.arange(0, len(success_list)), Y=np.array(success_list))
        # viz.line(X=x, Y=y, win='my_win')

        # Tensorboard
        if epoch is not None:
            self.scalar_summary('performance', (y - bp_data['y1']).sum(), epoch+1)





    ############################################################################
                            # TENSORBOARD LOGGER FUNCTIONS #
    ############################################################################
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
