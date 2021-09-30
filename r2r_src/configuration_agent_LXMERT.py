import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
from torch.functional import split
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from env_vision import R2RBatch
from utils import padding_idx, add_idx, Tokenizer, glove_embedding
import utils
import model
import param
from param import args
from collections import defaultdict
import encoder
from transformers import LxmertModel



class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents
    
    def write_results(self):
        output = [{'instr_id':k, 'trajectory': v} for k,v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))   # If iters is not none, shuffle the env batch
        self.losses = []
        self.results = {}
        # We rely on env showing the entire batch before repeating anything
        looped = False
        self.loss = 0
        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                for traj in self.rollout(**kwargs):
                    self.loss = 0
                    self.results[traj['instr_id']] = traj['path']
        else:   # Do a full round
            while True:
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.loss = 0
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
      'left': (0,-1, 0), # left
      'right': (0, 1, 0), # right
      'up': (0, 0, 1), # up
      'down': (0, 0,-1), # down
      'forward': (1, 0, 0), # forward
      '<end>': (0, 0, 0), # <end>
      '<start>': (0, 0, 0), # <start>
      '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size

        # Models
        enc_hidden_size = args.rnn_dim//2 if args.bidir else args.rnn_dim
        # self.encoder = model.EncoderLSTM(tok.vocab_size(), args.wemb, enc_hidden_size, padding_idx,
        #                                  args.dropout, bidirectional=args.bidir).cuda()

        encoder_kwargs = {
        'vocab_size': tok.vocab_size(),
        'embedding_size': args.word_embedding_size,
        'hidden_size': args.rnn_hidden_size,
        'padding_idx': padding_idx,
        'dropout_ratio': args.rnn_dropout,
        'bidirectional': args.bidirectional == 1,
        'num_layers': args.rnn_num_layers
    }
        self.encoder = encoder.EncoderBERT(**encoder_kwargs).cuda()
        self.decoder =  nn.DataParallel(model.ConfigurationLXMERTDecoder(args.aemb, args.rnn_dim, args.dropout, feature_size=self.feature_size + args.angle_feat_size).cuda())
        self.lxmert_model =  nn.DataParallel(LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased', cache_dir='/VL/space/zhan1624/lxmert-base-uncased').cuda())
        self.critic =  nn.DataParallel(model.Critic().cuda())
        self.models = (self.encoder, self.decoder, self.critic)

        # Optimizers
        self.encoder_optimizer = args.optimizer(self.encoder.parameters(), lr=args.lr)
        self.decoder_optimizer = args.optimizer(self.decoder.parameters(), lr=args.lr)
        self.critic_optimizer = args.optimizer(self.critic.parameters(), lr=args.lr)
        self.optimizers = (self.encoder_optimizer, self.decoder_optimizer, self.critic_optimizer)

        # Evaluations
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, size_average=False)

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
        self.candidate = {}

       
        # Glove feature of landmarks

        

        # Different spatial features
        self.motion_indicator_feature = self.get_motion_indicator_feature(args.train_motion_indi_path, args.val_seen_motion_indi_path, \
                                            args.val_unseen_motion_indi_path)
        self.landmark_feature = self.get_landmark_feature(args.train_landmark_path, args.val_seen_landmark_path, \
                                                        args.val_unseen_landmark_path)
        

    def softmax(self, x):
        s = torch.exp(x)
        return s / torch.sum(s, dim=1, keepdim=True)

    def crossEntropy(self, logits, y_true):
        c = -torch.log(logits.gather(1, y_true.reshape(-1, 1)))
        return torch.sum(c)
  
    def _sort_batch(self, obs):
        ''' Extract instructions from a list of observations and sort by descending
            sequence length (to enable PyTorch packing). '''

        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        seq_lengths = np.argmax(seq_tensor == padding_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]     # Full length

        seq_tensor = torch.from_numpy(seq_tensor)
        seq_lengths = torch.from_numpy(seq_lengths)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)       # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor == padding_idx)[:,:seq_lengths[0]]    # seq_lengths[0] is the Maximum length

        return Variable(sorted_tensor, requires_grad=False).long().cuda(), \
               mask.bool().cuda(),  \
               list(seq_lengths), list(perm_idx)

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.zeros((len(obs), args.views, self.feature_size + args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature'] 
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()
    
    
    def get_motion_indicator_feature(self, train_motion_indi_dir,val_seen_motion_indi_dir,val_unseen_motion_indi_dir, test_motion_indi_dir=None):
        motion_indicator_dict = {}
        motion_indicator_dict1 = np.load(train_motion_indi_dir, allow_pickle=True).item()
        motion_indicator_dict2 = np.load(val_seen_motion_indi_dir, allow_pickle=True).item()
        motion_indicator_dict3 = np.load(val_unseen_motion_indi_dir, allow_pickle=True).item()

        motion_indicator_dict.update(motion_indicator_dict1)
        motion_indicator_dict.update(motion_indicator_dict2)
        motion_indicator_dict.update(motion_indicator_dict3)
        return motion_indicator_dict
    
    def get_landmark_feature(self, train_landmark_dir, val_seen_landmark_dir, val_unseen_landmark_dir, test_landmark_dir=None):
        landmark_dict = {}
        landmark_dict1 = np.load(train_landmark_dir, allow_pickle=True).item()
        landmark_dict2 = np.load(val_seen_landmark_dir, allow_pickle=True).item()
        landmark_dict3 = np.load(val_unseen_landmark_dir, allow_pickle=True).item()

        landmark_dict.update(landmark_dict1)
        landmark_dict.update(landmark_dict2)
        landmark_dict.update(landmark_dict3)

        return landmark_dict

    def cartesian_product(self,x,y):
        xs = x.size()
        ys = y.size()
        assert xs[0] == ys[0]
        # torch cat is not broadcasting, do repeat manually
        xx = x.view(xs[0], xs[1], 1, xs[2]).repeat(1, 1, ys[1], 1)
        yy = y.view(ys[0], 1, ys[1], ys[2]).repeat(1, xs[1], 1, 1)
        return torch.cat([xx, yy], dim=-1)

    def _candidate_variable1(self, obs):
        object_num = 36
        feature_size1 = 2048
        object_rel = 6

        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]       # +1 is for the end
        max_candidate_leng = max(candidate_leng)
        candidate_obj_img_feat = np.zeros((len(obs), max_candidate_leng, object_num, feature_size1), dtype=np.float32)
        object_mask = np.zeros((len(obs), max_candidate_leng, object_num))
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + args.angle_feat_size), dtype=np.float32)
        object_index = np.zeros((len(obs), max_candidate_leng),dtype=np.int64)
        object_relation = np.zeros((len(obs), max_candidate_leng, object_rel),dtype=np.float32)
        object_boxes = np.zeros((len(obs), max_candidate_leng, object_num, 4),dtype=np.float32)
        object_text = []
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            tmp_object_text = [['']*object_num]*max_candidate_leng
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']        # Image feat
                if isinstance(c['obj_feat'], tuple):
                    c['obj_feat'] = c['obj_feat'][0]
                if isinstance(c['obj_mask'], tuple):
                    c['obj_mask'] = c['obj_mask'][0]
                candidate_obj_img_feat[i,j,:] = c['obj_feat']
                object_mask[i,j] = c['obj_mask']
                object_index[i,j] = c['pointId']
                object_relation[i,j,:] = c['obj_rel']
                object_boxes[i,j,:] = c['obj_boxes']
                tmp_object_text[j] = c['obj_text']
            object_text.append(tmp_object_text)
       
        return torch.from_numpy(candidate_feat).cuda(), candidate_leng, torch.from_numpy(candidate_obj_img_feat).cuda(), torch.from_numpy(object_mask).cuda(), torch.from_numpy(object_index).cuda(),\
             torch.from_numpy(object_relation).cuda(), object_text, torch.from_numpy(object_boxes).cuda()


    def distribute_feature(self, interval_len, pre_feature, feature, candidate_leng):
        pre_feature[0: interval_len] =  feature[0*interval_len:1*interval_len]
        pre_feature[candidate_leng:candidate_leng+interval_len] =  feature[1*interval_len:2*interval_len]
        pre_feature[2*candidate_leng:2*candidate_leng+interval_len] =  feature[2*interval_len:3*interval_len]
    
    def elevation_index(self, index):
        elevation_level = index % 12
        bottom_index = elevation_level
        middle_index = elevation_level + 12
        top_index = elevation_level + 12 + 12
        return bottom_index, middle_index, top_index
    

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), args.angle_feat_size), np.float32)
        
        for i, ob in enumerate(obs):
            input_a_t[i] = utils.angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)      # Image features from obs
        candidate_feat, candidate_leng, candidate_obj_text_feat, object_mask, object_index, object_relation, object_text, object_boxes = self._candidate_variable1(obs)

        return input_a_t, f_t, candidate_feat, candidate_obj_text_feat, object_mask, candidate_leng, object_index, object_relation, \
            object_text, object_boxes
    

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view 
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, idx, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:                       # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))
        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12   # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:    # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def sim_matrix(self, a, b, eps=1e-8):
        a_n, b_n = torch.norm(a,dim=-1,keepdim=True), torch.norm(b,dim=-1,keepdim=True)
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        sim_mt = torch.einsum("bijk,bijfk->bijf", a_norm, b_norm)
        return sim_mt

    def rollout(self, train_ml=None, train_rl=True, reset=True, speaker=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        if reset:
            # Reset env
            obs = np.array(self.env.reset())
        else:
            obs = np.array(self.env._get_obs())

        batch_size = len(obs)

        if speaker is not None:         # Trigger the self_train mode!
            noise = self.decoder.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            # Create fake environments with the generated instruction
            boss = np.ones((batch_size, 1), np.int64) * self.tok.word_to_index['<BOS>']  # First word is <BOS>
            insts = np.concatenate((boss, insts), 1)
            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('instr_encoding')
                datum['instructions'] = self.tok.decode_sentence(inst)
                datum['instr_encoding'] = inst
            obs = np.array(self.env.reset(batch))

        # Reorder the language input for the encoder (do not ruin the original code)
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        perm_obs = obs[perm_idx]
        sentence = []
        config_num_list = []
        instr_id_list = []

        for ob_id, each_ob in enumerate(perm_obs):
            instr_id_list.append(each_ob['instr_id']) 
            config_num_list.append(len(each_ob['configurations']))
            sentence.append(each_ob['instructions'])
        max_config_num = max(config_num_list)

        ### Collect all pre-processed landmarks and motion_indicator
        landmark_text = []
        motion_indicator_tensor = torch.zeros(batch_size, max_config_num, args.text_dimension).cuda()
        for ob_id, each_ob in enumerate(perm_obs):
            tmp_landmark_text = []
            for config_id, each_config in enumerate(each_ob['configurations']):
                # Obtain motion_indicator representation
                motion_indi = self.motion_indicator_feature[each_ob['instr_id']+"_"+str(config_id)][0]
                motion_indi = motion_indi.split(' ')
                try: # Deal with cases without motion indicators
                    motion_indi_tmp_tensor = torch.tensor(np.stack([self.encoder.glove_vector[self.encoder.glove_indexer.add_and_get_index(each_m)] for each_m in motion_indi if each_m]))
                except IndexError:
                    motion_indi_tmp_tensor = torch.zeros(300)
                motion_indicator_tensor[ob_id, config_id,:] = torch.mean(motion_indi_tmp_tensor,dim=0)
                # Obtain landmark list in each instruction
                tmp_landmark_tuple = self.landmark_feature[each_ob['instr_id']+"_"+str(config_id)][0]
                for _, each_landmark in enumerate(tmp_landmark_tuple):
                    if each_landmark[0]:
                        tmp_landmark_text.append(each_landmark[0])
            landmark_text.append(tmp_landmark_text)

        ### Obtain bert representation of the instructions
        max_seq_len = 80
        tmp_ctx, h_t, c_t, tmp_ctx_mask, split_index, original_embeds, landmark_id_list = self.encoder(sentence, seq_len=max_seq_len, landmark_input=landmark_text)
        
        ### Split instruction representation into configuration representations
        token_num = max([each_split[-1] for each_split in split_index])
        max_config_num = max(config_num_list) # There are cases that are statisfied with this assertion statement when the length of the sentence larger than 80. So here we just heuritically set all equals max_config_num
        bert_ctx = torch.zeros(batch_size, max_config_num, token_num, 512, device = tmp_ctx.device)
        bert_ctx_mask = torch.zeros(batch_size, max_config_num, token_num, device = tmp_ctx.device)
        bert_cls = torch.zeros(batch_size, max_config_num, 512, device = tmp_ctx.device)
        bert_cls_mask = torch.zeros(batch_size, max_config_num, device = tmp_ctx.device)

        for ob_id, each_index_list in enumerate(split_index):
            start = 0
            for list_id, each_index in enumerate(each_index_list):
                end = each_index
                bert_ctx[ob_id,list_id,0:end-start,:] = tmp_ctx[ob_id,start:end,:]
                bert_ctx_mask[ob_id, list_id,0:end-start] = tmp_ctx_mask[ob_id,start:end]
                bert_cls[ob_id, list_id, :] = tmp_ctx[ob_id, each_index,:]
                bert_cls_mask[ob_id, list_id] = 1
                start = end + 1
        
        ### allocate landmark id to the configuration id
        def get_landmark_repr(config_split_index, landmark_split_index):
            """
            :param config_split_index: split id for configurations
            :param landmark_split_indx: split id for landmarks
            :param glove_true whether use glove representation. 

            :return: return glove representation, else return landmark id.
            """
            split_list1 = []
            split_list2 = []
            for split_id, each_split in enumerate(config_split_index):
                tmp_list1 = []
                tmp_list2 = []
                # Allocate landmark_split_index into config_split_index
                for list_id1, each_index1 in enumerate(each_split):
                    tmp_tmp_list1 = []
                    tmp_tmp_list2 = []
                    for land_key, land_value in landmark_split_index[split_id].items():
                        if list_id1 == 0:
                            if land_key <= each_index1:
                                tmp_tmp_list1.append(land_key)
                                tmp_tmp_list2.append(land_value)
                        else:
                            if land_key >= each_split[list_id1-1] and land_key <= each_index1:
                                tmp_tmp_list1.append(land_key)
                                tmp_tmp_list2.append(land_value)
                    tmp_list1.append(tmp_tmp_list1)
                    tmp_list2.append(tmp_tmp_list2)
                split_list1.append(tmp_list1)
                split_list2.append(tmp_list2)
            return split_list1, split_list2
            
        ### Obtain landmark bert representation
        max_landmark_num =max([len(i) for i in landmark_id_list])
        landmark_split_id_list, landmark_split_vector = get_landmark_repr(split_index, landmark_id_list)
        landmark_mask = torch.zeros(batch_size, max_config_num, max_landmark_num)-1
        if args.using_landmark_bert_repre:
            landmark_repr = torch.zeros(batch_size, max_config_num, max_landmark_num, 768) # Dimension for bert representation is 768
            landmark_split_id_list = get_landmark_repr(split_index, landmark_id_list)
            for split_id, each_split in enumerate(split_index):
                for list_id, _ in enumerate(each_split):
                    if landmark_split_id_list[split_id][list_id]: 
                        tmp_len = len(landmark_split_id_list[split_id][list_id])          
                        landmark_repr[split_id, list_id, :tmp_len, :] = original_embeds[split_id, landmark_split_id_list[split_id][list_id],:]
                        landmark_mask[split_id, list_id, :tmp_len] = torch.tensor([1]*tmp_len)

        ### Obtain landmark glove representation
        else:
            landmark_repr = torch.zeros(batch_size, max_config_num, max_landmark_num, 300) # Dimension for glove representation is 300
            for split_id, each_split in enumerate(split_index):
                for list_id, _ in enumerate(each_split):
                    if landmark_split_vector[split_id][list_id]:
                        tmp_len = len(landmark_split_vector[split_id][list_id]) 
                        landmark_repr[split_id, list_id, :tmp_len, :] = torch.tensor(np.stack(landmark_split_vector[split_id][list_id],axis=0))
                        landmark_mask[split_id, list_id, :tmp_len] = torch.tensor([1]*tmp_len)

        ### Using the representation of motion indicators and landmarks to attend configuration representation
        motion_indicator_tensor = motion_indicator_tensor.cuda()
        landmark_repr = landmark_repr.cuda()
        landmark_mask = landmark_mask.cuda()

        atten_ctx1, _ = self.encoder.sf(bert_cls, bert_cls_mask, bert_ctx, bert_ctx_mask)
        atten_ctx2, _ = self.encoder.sf(motion_indicator_tensor, bert_cls_mask, bert_ctx, bert_ctx_mask)
        atten_ctx3, _ = self.encoder.sf(torch.mean(landmark_repr, dim=2), bert_cls_mask, bert_ctx, bert_ctx_mask)
        #ctx = torch.cat([attend_ctx, torch.mean(landmark_object_feature, dim=2), motion_indicator_tensor], dim=2)   
        ctx = torch.mean(torch.stack([atten_ctx1, atten_ctx2, atten_ctx3],dim=2), dim=2)
        ctx_mask = (bert_cls_mask==0).byte()

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(perm_obs):   # The init distance from the view point to the target
            last_dist[i] = ob['distance']

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in perm_obs]

        # For test result submission
        visited = [set() for _ in perm_obs]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards = []
        hidden_states = []
        policy_log_probs = []
        masks = []
        entropys = []
        ml_loss = 0.

        h1 = h_t

        #state attention
        s0 = torch.zeros(batch_size, max_config_num, requires_grad=False).cuda()
        r0 = torch.zeros(batch_size,2, requires_grad=False).cuda()
        s0[:,0] = 1
        r0[:,0] = 1
        ctx_attn = s0
        top_N = 3

        for t in range(self.episode_len):
            #input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
            #candidate_obj_text_feat = self.get_cand_object_feat(perm_obs)

            input_a_t, f_t, candi_feat, candi_obj_img_feat, candi_object_mask, candidate_leng,  object_index, object_relation, \
            candi_object_text, candi_obj_boxes = self.get_input_feat(perm_obs)
                 
            if speaker is not None:       # Apply the env drop mask to the feat
                candi_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise

            r_t = r0 if t==0 else None
            #landamrk similarity 

            candidate_mask = utils.length2mask(candidate_leng)

            new_candidate_leng = list(map(lambda x:x-1,candidate_leng))
            new_candidate_mask = utils.length2mask(new_candidate_leng, size=max(candidate_leng))

            candi_image_num = candi_obj_img_feat.shape[1]

            new_land_text = []
            for ob_id, each_sent in enumerate(sentence):
                tmp_land_list = [each_sent] * candi_image_num
                new_land_text += tmp_land_list

            ## Obtain lxmert representation of the instruction and objects
            tmp_original_embeds = original_embeds.unsqueeze(1).expand(batch_size,candi_image_num, max_seq_len,768).contiguous().view(-1,max_seq_len,768)
            candi_lx_lang, candi_lx_obj, _ = self.lxmert_model(inputs_embeds=tmp_original_embeds, visual_feats=candi_obj_img_feat.view(-1,36,2048), visual_pos=candi_obj_boxes.view(-1,36,4))
            
            ### Obtain lxmert representation of landmark
            lxmert_landmark_repr = torch.zeros(batch_size, candi_image_num, max_config_num, max_landmark_num, 768).cuda()
            for split_id, each_split in enumerate(split_index):
                obj_start = split_id * candi_image_num
                obj_end = (split_id+1) * candi_image_num
                for list_id, _ in enumerate(each_split):
                    if landmark_split_id_list[split_id][list_id]:
                        lxmert_landmark_repr[split_id, :, list_id, :len(landmark_split_id_list[split_id][list_id]), :] = candi_lx_lang[obj_start:obj_end, landmark_split_id_list[split_id][list_id],:]


            lxmert_landmark_repr = lxmert_landmark_repr.view(batch_size, candi_image_num, -1, 768)
            candi_lx_obj = candi_lx_obj.view(batch_size, candi_image_num, -1, 36, 768)
            sim_matrix = torch.max(self.sim_matrix(lxmert_landmark_repr, candi_lx_obj).view(batch_size, candi_image_num, -1, 36), dim=-1)
            
        
            h_t, c_t, logit, h1, ctx_attn = self.decoder(input_a_t, f_t, candi_feat, h_t, h1, c_t,
                                                        ctx, t, ctx_attn, r_t, ctx_mask, candi_object_mask=candi_object_mask, candidate_mask = new_candidate_mask, 
                                                        candi_landmark = landmark_repr, candidate_obj_feat = candi_obj_img_feat, landmark_mask=landmark_mask, 
                                                        object_index =object_index, sim_matrix = sim_matrix, already_dropfeat=(speaker is not None))

       

            hidden_states.append(h_t)
            # Mask outputs where agent can't move forward
            # Here the logit is [b, max_candidate]
            
            if args.submit:     # Avoding cyclic path
                for ob_id, ob in enumerate(perm_obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            

            logit.masked_fill_(candidate_mask, -float('inf'))

            # Supervised training
            target = self._teacher_action(perm_obs, ended)
            ml_loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax': 
                _, a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
                log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
                policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))   # Gather the log_prob for each batch
            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)    # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'].append(c.entropy().sum().item())      # For log
                entropys.append(c.entropy())                                # For optimization
                a_t = c.sample().detach()
                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i]-1) or next_id == args.ignoreid or ended[i]:    # The last action is <end>
                    cpu_a_t[i] = -1             # Change the <end> and ignore action to -1

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, perm_obs, perm_idx, traj)
            obs = np.array(self.env._get_obs())
            perm_obs = obs[perm_idx]                    # Perm the obs for the resu

            # Calculate the mask and reward
            dist = np.zeros(batch_size, np.float32)
            reward = np.zeros(batch_size, np.float32)
            mask = np.ones(batch_size, np.float32)
            for i, ob in enumerate(perm_obs):
                dist[i] = ob['distance']
                if ended[i]:            # If the action is already finished BEFORE THIS ACTION.
                    reward[i] = 0.
                    mask[i] = 0.
                else:       # Calculate the reward
                    action_idx = cpu_a_t[i]
                    if action_idx == -1:        # If the action now is end
                        if dist[i] < 3:         # Correct
                            reward[i] = 2.
                        else:                   # Incorrect
                            reward[i] = -2.
                    else:                       # The action is not end
                        reward[i] = - (dist[i] - last_dist[i])      # Change of distance
                        if reward[i] > 0:                           # Quantification
                            reward[i] = 1
                        elif reward[i] < 0:
                            reward[i] = -1
                        else:
                            raise NameError("The action doesn't change the move")
            rewards.append(reward)
            masks.append(mask)
            last_dist[:] = dist

            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all(): 
                break

        if train_rl:
            # Last action in A2C
            #
            # input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
                 
            input_a_t, f_t, candi_feat, candi_obj_img_feat, candi_object_mask, candidate_leng,  object_index, object_relation, \
            candi_object_text, candi_obj_boxes = self.get_input_feat(perm_obs)
                 
            if speaker is not None:       # Apply the env drop mask to the feat
                candi_feat[..., :-args.angle_feat_size] *= noise
                f_t[..., :-args.angle_feat_size] *= noise

            r_t = r0 if t==0 else None
            #landamrk similarity 

            candidate_mask = utils.length2mask(candidate_leng)

            new_candidate_leng = list(map(lambda x:x-1,candidate_leng))
            new_candidate_mask = utils.length2mask(new_candidate_leng, size=max(candidate_leng))

            candi_image_num = candi_obj_img_feat.shape[1]

            new_land_text = []
            for ob_id, each_sent in enumerate(sentence):
                tmp_land_list = [each_sent] * candi_image_num
                new_land_text += tmp_land_list

            ## Obtain lxmert representation of the instruction and objects
            tmp_original_embeds = original_embeds.unsqueeze(1).expand(batch_size,candi_image_num, max_seq_len,768).contiguous().view(-1,max_seq_len,768)
            candi_lx_lang, candi_lx_obj, _ = self.lxmert_model(inputs_embeds=tmp_original_embeds, visual_feats=candi_obj_img_feat.view(-1,36,2048), visual_pos=candi_obj_boxes.view(-1,36,4))
            
            ### Obtain lxmert representation of landmark
            lxmert_landmark_repr = torch.zeros(batch_size, candi_image_num, max_config_num, max_landmark_num, 768).cuda()
            for split_id, each_split in enumerate(split_index):
                obj_start = split_id * candi_image_num
                obj_end = (split_id+1) * candi_image_num
                for list_id, _ in enumerate(each_split):
                    if landmark_split_id_list[split_id][list_id]:
                        lxmert_landmark_repr[split_id, :, list_id, :len(landmark_split_id_list[split_id][list_id]), :] = candi_lx_lang[obj_start:obj_end, landmark_split_id_list[split_id][list_id],:]


            lxmert_landmark_repr = lxmert_landmark_repr.view(batch_size, candi_image_num, -1, 768)
            candi_lx_obj = candi_lx_obj.view(batch_size, candi_image_num, -1, 36, 768)
            sim_matrix = torch.max(self.sim_matrix(lxmert_landmark_repr, candi_lx_obj).view(batch_size, candi_image_num, -1, 36), dim=-1)
            
        
            last_h_, _, _, _, _ = self.decoder(input_a_t, f_t, candi_feat, h_t, h1, c_t,
                                                        ctx, t, ctx_attn, r_t, ctx_mask, candi_object_mask=candi_object_mask, candidate_mask = new_candidate_mask, 
                                                        candi_landmark = landmark_repr, candidate_obj_feat = candi_obj_img_feat, landmark_mask=landmark_mask, 
                                                        object_index =object_index, sim_matrix = sim_matrix, already_dropfeat=(speaker is not None))

            rl_loss = 0.

            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ = self.critic(last_h_).detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)
            total = 0
            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = self.critic(hidden_states[t])
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                rl_loss += (((r_ - v_) ** 2) * mask_).sum() * 0.5     # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()
                self.logs['critic_loss'].append((((r_ - v_) ** 2) * mask_).sum().item())

                total = total + np.sum(masks[t])
            self.logs['total'].append(total)

            # Normalize the loss function
            if args.normalize_loss == 'total':
                rl_loss /= total
            elif args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert args.normalize_loss == 'none'

            self.loss += rl_loss

        if train_ml is not None:
            self.loss += ml_loss * train_ml / batch_size

        if type(self.loss) is int:  # For safety, it will be activated if no losses are added
            self.losses.append(0.)
        else:
            self.losses.append(self.loss.item() / self.episode_len)    # This argument is useless.

        return traj

    def _dijkstra(self):
        """
        The dijkstra algorithm.
        Was called beam search to be consistent with existing work.
        But it actually finds the Exact K paths with smallest listener log_prob.
        :return:
        [{
            "scan": XXX
            "instr_id":XXX,
            'instr_encoding": XXX
            'dijk_path': [v1, v2, ..., vn]      (The path used for find all the candidates)
            "paths": {
                    "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        }]
        """
        def make_state_id(viewpoint, action):     # Make state id
            return "%s_%s" % (viewpoint, str(action))
        def decompose_state_id(state_id):     # Make state id
            viewpoint, action = state_id.split("_")
            action = int(action)
            return viewpoint, action

        # Get first obs
        obs = self.env._get_obs()

        # Prepare the state id
        batch_size = len(obs)
        results = [{"scan": ob['scan'],
                    "instr_id": ob['instr_id'],
                    "instr_encoding": ob["instr_encoding"],
                    "dijk_path": [ob['viewpoint']],
                    "paths": []} for ob in obs]

        # Encoder
        seq, seq_mask, seq_lengths, perm_idx = self._sort_batch(obs)
        recover_idx = np.zeros_like(perm_idx)
        for i, idx in enumerate(perm_idx):
            recover_idx[idx] = i
        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        ctx, h_t, c_t, ctx_mask = ctx[recover_idx], h_t[recover_idx], c_t[recover_idx], seq_mask[recover_idx]    # Recover the original order

        # Dijk Graph States:
        id2state = [
            {make_state_id(ob['viewpoint'], -95):
                 {"next_viewpoint": ob['viewpoint'],
                  "running_state": (h_t[i], h_t[i], c_t[i]),
                  "location": (ob['viewpoint'], ob['heading'], ob['elevation']),
                  "feature": None,
                  "from_state_id": None,
                  "score": 0,
                  "scores": [],
                  "actions": [],
                  }
             }
            for i, ob in enumerate(obs)
        ]    # -95 is the start point
        visited = [set() for _ in range(batch_size)]
        finished = [set() for _ in range(batch_size)]
        graphs = [utils.FloydGraph() for _ in range(batch_size)]        # For the navigation path
        ended = np.array([False] * batch_size)

        # Dijk Algorithm
        for _ in range(300):
            # Get the state with smallest score for each batch
            # If the batch is not ended, find the smallest item.
            # Else use a random item from the dict  (It always exists)
            smallest_idXstate = [
                max(((state_id, state) for state_id, state in id2state[i].items() if state_id not in visited[i]),
                    key=lambda item: item[1]['score'])
                if not ended[i]
                else
                next(iter(id2state[i].items()))
                for i in range(batch_size)
            ]

            # Set the visited and the end seqs
            for i, (state_id, state) in enumerate(smallest_idXstate):
                assert (ended[i]) or (state_id not in visited[i])
                if not ended[i]:
                    viewpoint, action = decompose_state_id(state_id)
                    visited[i].add(state_id)
                    if action == -1:
                        finished[i].add(state_id)
                        if len(finished[i]) >= args.candidates:     # Get enough candidates
                            ended[i] = True

            # Gather the running state in the batch
            h_ts, h1s, c_ts = zip(*(idXstate[1]['running_state'] for idXstate in smallest_idXstate))
            h_t, h1, c_t = torch.stack(h_ts), torch.stack(h1s), torch.stack(c_ts)

            # Recover the env and gather the feature
            for i, (state_id, state) in enumerate(smallest_idXstate):
                next_viewpoint = state['next_viewpoint']
                scan = results[i]['scan']
                from_viewpoint, heading, elevation = state['location']
                self.env.env.sims[i].newEpisode(scan, next_viewpoint, heading, elevation) # Heading, elevation is not used in panoramic
            obs = self.env._get_obs()

            # Update the floyd graph
            # Only used to shorten the navigation length
            # Will not effect the result
            for i, ob in enumerate(obs):
                viewpoint = ob['viewpoint']
                if not graphs[i].visited(viewpoint):    # Update the Graph
                    for c in ob['candidate']:
                        next_viewpoint = c['viewpointId']
                        dis = self.env.distances[ob['scan']][viewpoint][next_viewpoint]
                        graphs[i].add_edge(viewpoint, next_viewpoint, dis)
                    graphs[i].update(viewpoint)
                results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], viewpoint))

            input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(obs)

            # Run one decoding step
            h_t, c_t, alpha, logit, h1 = self.decoder(input_a_t, f_t, candidate_feat,
                                                      h_t, h1, c_t,
                                                      ctx, ctx_mask,
                                                      False)

            # Update the dijk graph's states with the newly visited viewpoint
            candidate_mask = utils.length2mask(candidate_leng)
            logit.masked_fill_(candidate_mask, -float('inf'))
            log_probs = F.log_softmax(logit, 1)                              # Calculate the log_prob here
            _, max_act = log_probs.max(1)

            for i, ob in enumerate(obs):
                current_viewpoint = ob['viewpoint']
                candidate = ob['candidate']
                current_state_id, current_state = smallest_idXstate[i]
                old_viewpoint, from_action = decompose_state_id(current_state_id)
                assert ob['viewpoint'] == current_state['next_viewpoint']
                if from_action == -1 or ended[i]:       # If the action is <end> or the batch is ended, skip it
                    continue
                for j in range(len(ob['candidate']) + 1):               # +1 to include the <end> action
                    # score + log_prob[action]
                    modified_log_prob = log_probs[i][j].detach().cpu().item() 
                    new_score = current_state['score'] + modified_log_prob
                    if j < len(candidate):                        # A normal action
                        next_id = make_state_id(current_viewpoint, j)
                        next_viewpoint = candidate[j]['viewpointId']
                        trg_point = candidate[j]['pointId']
                        heading = (trg_point % 12) * math.pi / 6
                        elevation = (trg_point // 12 - 1) * math.pi / 6
                        location = (next_viewpoint, heading, elevation)
                    else:                                                 # The end action
                        next_id = make_state_id(current_viewpoint, -1)    # action is -1
                        next_viewpoint = current_viewpoint                # next viewpoint is still here
                        location = (current_viewpoint, ob['heading'], ob['elevation'])

                    if next_id not in id2state[i] or new_score > id2state[i][next_id]['score']:
                        id2state[i][next_id] = {
                            "next_viewpoint": next_viewpoint,
                            "location": location,
                            "running_state": (h_t[i], h1[i], c_t[i]),
                            "from_state_id": current_state_id,
                            "feature": (f_t[i].detach().cpu(), candidate_feat[i][j].detach().cpu()),
                            "score": new_score,
                            "scores": current_state['scores'] + [modified_log_prob],
                            "actions": current_state['actions'] + [len(candidate)+1],
                        }

            # The active state is zero after the updating, then setting the ended to True
            for i in range(batch_size):
                if len(visited[i]) == len(id2state[i]):     # It's the last active state
                    ended[i] = True

            # End?
            if ended.all():
                break

        # Move back to the start point
        for i in range(batch_size):
            results[i]['dijk_path'].extend(graphs[i].path(results[i]['dijk_path'][-1], results[i]['dijk_path'][0]))
        """
            "paths": {
                "trajectory": [viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }
        """
        # Gather the Path
        for i, result in enumerate(results):
            assert len(finished[i]) <= args.candidates
            for state_id in finished[i]:
                path_info = {
                    "trajectory": [],
                    "action": [],
                    "listener_scores": id2state[i][state_id]['scores'],
                    "listener_actions": id2state[i][state_id]['actions'],
                    "visual_feature": []
                }
                viewpoint, action = decompose_state_id(state_id)
                while action != -95:
                    state = id2state[i][state_id]
                    path_info['trajectory'].append(state['location'])
                    path_info['action'].append(action)
                    path_info['visual_feature'].append(state['feature'])
                    state_id = id2state[i][state_id]['from_state_id']
                    viewpoint, action = decompose_state_id(state_id)
                state = id2state[i][state_id]
                path_info['trajectory'].append(state['location'])
                for need_reverse_key in ["trajectory", "action", "visual_feature"]:
                    path_info[need_reverse_key] = path_info[need_reverse_key][::-1]
                result['paths'].append(path_info)

        return results

    def beam_search(self, speaker):
        """
        :param speaker: The speaker to be used in searching.
        :return:
        {
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                "action": [act_1, act_2, ..., ],
                "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                "speaker_scores": [log_prob_word1, log_prob_word2, ..., ],
            }]
        }
        """
        self.env.reset()
        results = self._dijkstra()
        """
        return from self._dijkstra()
        [{
            "scan": XXX
            "instr_id":XXX,
            "instr_encoding": XXX
            "dijk_path": [v1, v2, ...., vn]
            "paths": [{
                    "trajectory": [viewoint_id0, viewpoint_id1, viewpoint_id2, ..., ],
                    "action": [act_1, act_2, ..., ],
                    "listener_scores": [log_prob_act1, log_prob_act2, ..., ],
                    "visual_feature": [(f1_step1, f2_step2, ...), (f1_step2, f2_step2, ...)
            }]
        }]
        """

        # Compute the speaker scores:
        for result in results:
            lengths = []
            num_paths = len(result['paths'])
            for path in result['paths']:
                assert len(path['trajectory']) == (len(path['visual_feature']) + 1)
                lengths.append(len(path['visual_feature']))
            max_len = max(lengths)
            img_feats = torch.zeros(num_paths, max_len, 36, self.feature_size + args.angle_feat_size)
            can_feats = torch.zeros(num_paths, max_len, self.feature_size + args.angle_feat_size)
            for j, path in enumerate(result['paths']):
                for k, feat in enumerate(path['visual_feature']):
                    img_feat, can_feat = feat
                    img_feats[j][k] = img_feat
                    can_feats[j][k] = can_feat
            img_feats, can_feats = img_feats.cuda(), can_feats.cuda()
            features = ((img_feats, can_feats), lengths)
            insts = np.array([result['instr_encoding'] for _ in range(num_paths)])
            seq_lengths = np.argmax(insts == self.tok.word_to_index['<EOS>'], axis=1)   # len(seq + 'BOS') == len(seq + 'EOS')
            insts = torch.from_numpy(insts).cuda()
            speaker_scores = speaker.teacher_forcing(train=True, features=features, insts=insts, for_listener=True)
            for j, path in enumerate(result['paths']):
                path.pop("visual_feature")
                path['speaker_scores'] = -speaker_scores[j].detach().cpu().numpy()[:seq_lengths[j]]
        return results

    def beam_search_test(self, speaker):
        self.encoder.eval()
        self.decoder.eval()
        self.critic.eval()

        looped = False
        self.results = {}
        while True:
            for traj in self.beam_search(speaker):
                if traj['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[traj['instr_id']] = traj
            if looped:
                break

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            self.critic.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.critic.eval()
        super(Seq2SeqAgent, self).test(iters)

    def zero_grad(self):
        self.loss = 0.
        self.losses = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()

    def accumulate_gradient(self, feedback='teacher', **kwargs):
        if feedback == 'teacher':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
        elif feedback == 'sample':
            self.feedback = 'teacher'
            self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
            self.feedback = 'sample'
            self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def optim_step(self):
        self.loss.backward()

        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.critic_optimizer.step()

    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        self.losses = []
        for iter in range(1, n_iters + 1):

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            self.loss = 0
            if feedback == 'teacher':
                self.feedback = 'teacher'
                self.rollout(train_ml=args.teacher_weight, train_rl=False, **kwargs)
            elif feedback == 'sample':
                if args.ml_weight != 0:
                    self.feedback = 'teacher'
                    self.rollout(train_ml=args.ml_weight, train_rl=False, **kwargs)
                self.feedback = 'sample'
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                assert False

            self.loss.backward()

            torch.nn.utils.clip_grad_norm(self.encoder.parameters(), 40.)
            torch.nn.utils.clip_grad_norm(self.decoder.parameters(), 40.)

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.critic_optimizer.step()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            model_keys = set(state.keys())
            load_keys = set(states[name]['state_dict'].keys())
            if model_keys != load_keys:
                print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
            state.update(states[name]['state_dict'])
            model.load_state_dict(state)
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("encoder", self.encoder, self.encoder_optimizer),
                     ("decoder", self.decoder, self.decoder_optimizer),
                     ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
            recover_state(*param)
        return states['encoder']['epoch'] - 1