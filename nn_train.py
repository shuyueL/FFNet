import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent

# data path and names.
feat_path = '/home/slan001/NEW/newVS0912/Tour20/FeaturePool/'
gt_path ='/home/slan001/NEW/newVS0912/Tour20/GTPool_new/gt_gen/'

# split 1
train_name = [ \
'AW1','AW2','AW3','AW4','AW5','AW6', \
'MP1','MP2','MP3','MP4','MP5','MP6', \
'TM1','TM2','TM3','TM4','TM5','TM6', \
'BF1','BF2','BF3','BF4','BF5', \
'SB1','SB2','SB3','SB4', \
'MC1','MC2','MC3','MC4','MC5','MC6','MC7','MC8', \
'AT1','AT2','AT3','AT4','AT5', \
'GB1','GB2','GB3','GB4','GB5', \
'ET1','ET2','ET3','ET4','ET5','ET6', \
'NC1','NC2','NC3','NC4','NC5','NC6', \
'TA1','TA2','TA3','TA4','TA5', \
'HM1','HM2','HM3','HM4','HM5', \
'CB1','CB2','CB3','CB4','CB5', \
'GM1','GM2','GM3','GM4', \
'BK1','BK2','BK3','BK4','BK5','BK6','BK7', \
'WP1','WP2','WP3','WP4', \
'CI1','CI2','CI3','CI4','CI5','CI6', \
'SH1','SH2','SH3','SH4','SH5','SH6','SH7','SH8', \
'PT1','PT2','PT3','PT4','PT5','PT6','PT7', \
'PC1','PC2','PC3','PC4','PC5']

train_num = 113

# define neural network layout
l1 = Layer(4096,400,'relu')
l2 = Layer(400,200,'relu')
l3 = Layer(200,100,'relu')
l4 = Layer(100,25,'linear')
layers = [l1,l2,l3,l4]
learning_rate = 0.0002
loss_type = 'mean_square'
opt_type = 'RMSprop'

# set Q learning parameters
batch_size = 128
exp_rate = 1
exp_low = 0.1
exp_decay = 0.00001
decay_rate = 0.8
max_eps = 1000
savepath = 'model_R3_A25_S1_1013/'
filename = 'Q_net_all_11_0_1000'

# define Q learning agent
agent = Agent(layers, batch_size, exp_rate,exp_low,exp_decay, learning_rate, decay_rate,savepath)

# Training process
for index in range(max_eps*train_num):

    current_eps = Episode(index,train_num, train_name, feat_path, gt_path)

    agent.data_init(current_eps)

    agent.episode_run()

    pos = 0
    true_pos = 0
    summ = 0
    for i in range(current_eps.get_size()):
        if current_eps.gt[0][i]==1:
            pos = pos+1
        if current_eps.gt[0][i]==1 and agent.selection[i]==1:
            true_pos = true_pos+1
	if agent.selection[i] ==1:
	    summ = summ+1
    recall = float(true_pos)/float(pos)
    precision = float(true_pos)/float(summ)
    fscore = 2*precision*recall/(precision+recall)
    if index%train_num == 112:
        print('episode:'+str(index/113)+', gt: '+str(pos)+', sum: '+str(summ)+', tp: '+ str(true_pos)+', r: '+str(recall)+', p: '+str(precision)+', f: '+str(fscore))

    agent.data_reset()

agent.save_model(filename)
