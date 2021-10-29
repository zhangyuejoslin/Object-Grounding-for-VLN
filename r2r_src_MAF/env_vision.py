''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('buildpy36')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args
from tqdm import tqdm

from utils import load_datasets, load_nav_graphs, Tokenizer, get_configurations, get_motion_indicator, get_landmark

csv.field_size_limit(sys.maxsize)

class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, pano_caffee=None, pano_caffee_text=None, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        self.configs = {}
      
        self.motion_indicator = {}
        self.landmark = {}
   
        if not name:
            configs = np.load(args.configpath+"configs_"+splits[0]+".npy", allow_pickle=True).item()
            self.configs.update(configs)
    
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for item in tqdm(load_datasets(splits)):
            # Split multiple instructions into separate entries
            for j,instr in enumerate(item['instructions']):
                if item['scan'] not in self.env.featurized_scans:   # For fast training
                    continue
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                #new_item['instr_id'] = str(item['path_id'])
                if args.configuration and not name:
                    #each_configuration_list = get_configurations(instr)
                    #self.configs[str(new_item['instr_id'])] = each_configuration_list
                    each_configuration_list = self.configs[str(new_item['instr_id'])]
                    # instr = "Exit closet, and walk past bed. Walk out open bedroom door, and wait at top of stair landing. "
                    # each_configuration_list = get_configurations(instr)
                    # self.configs[str(new_item['instr_id'])] = each_configuration_list

                    
                    #for config_id, each_c in enumerate(each_configuration_list):
                        #self.motion_indicator[str(new_item['instr_id']) + "_" + str(config_id)] = get_motion_indicator(each_c)
                        #self.landmark[str(new_item['instr_id']) + "_" + str(config_id)] = get_landmark(each_c, whether_root=True)
                

                    '''
                    for config_id, each_c in enumerate(each_configuration_list):
                        self.motion_indicator[str(item['path_id']) + "_" + str(config_id)] = get_motion_indicator(each_c)
                        self.landmark[str(item['path_id']) + "_" + str(config_id)] = get_landmark(each_c)
                    '''
                    new_item['configurations'] = each_configuration_list
                    configuration_length = len(each_configuration_list)
                    tmp_str = " Quan ".join(each_configuration_list) + " Quan"
                    new_item['instructions'] = tmp_str
                    if configuration_length:
                        self.data.append((len(new_item['configurations']), new_item)) 

                    if tokenizer:
                        if 'instr_encoding' not in item:  # we may already include 'instr_encoding' when generating synthetic instructions       
                            new_item['instr_encoding'] = tokenizer.encode_sentence(tmp_str)

                else:
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                scans.append(item['scan'])

    
        # np.save(f"/home/joslin/R2R-EnvDrop/r2r_src/R4R_components/configs/configs_{splits[0]}.npy", self.configs)
        # np.save(f"/home/joslin/R2R-EnvDrop/r2r_src/R4R_components/motion_indicator/motion_indicator_{splits[0]}.npy", self.motion_indicator)
        # np.save(f"/home/joslin/R2R-EnvDrop/r2r_src/R4R_components/landmarks/landmark_{splits[0]}.npy", self.landmark)
        
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name
        self.pano_caffee = pano_caffee
        self.pano_caffee_text = pano_caffee_text
        self.scans = set(scans)
        self.splits = splits
        
        
        
        #self.data.sort(key=lambda x: x[0])
        if not name:
            self.data = list(map(lambda item:item[1], self.data))
       
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        
        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        
        def get_relative_position(loc_heading, base_heading, loc_elevation):
            left, right, front, back, up, down = 0, 0, 0, 0, 0, 0
            if abs(loc_heading) >=  math.pi/180*180:
                if loc_heading > 0:
                    loc_heading = loc_heading - math.pi/180*360      
                else:
                    loc_heading = loc_heading + math.pi/180*360

            if loc_heading < 0:
                left = 1
                if loc_heading > -math.pi/180 * 90:
                    front = 1
                else: 
                    back = 1
            else:
                right = 1
                if loc_heading < math.pi/180 * 90:
                    front = 1
                else:
                    back = 1
            
            if loc_elevation < -math.pi/180 * 30:
                down = 1
            elif loc_elevation >= math.pi/180 * 30:
                up = 1
            
            return [left, right, front, back, up, down]
        
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)
                
                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    relative_position = get_relative_position(loc_heading, base_heading, loc_elevation)

                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'obj_feat': self.pano_caffee[scanId][viewpointId][ix],
                            'obj_boxes': self.pano_caffee_text[scanId][viewpointId][ix]['boxes'],
                            'obj_mask': self.pano_caffee_text[scanId][viewpointId][ix]['text_mask'],
                            'obj_text':self.pano_caffee_text[scanId][viewpointId][ix]['text'],
                            'obj_rel': relative_position
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                # if c_new['scanId']+"_"+c_new['viewpointId'] in self.buffered_state_dict:
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new['obj_feat'] = self.pano_caffee[scanId][c_new['viewpointId']][ix]
                c_new['obj_boxes'] = self.pano_caffee_text[scanId][c_new['viewpointId']][ix]['boxes']
                c_new['obj_mask']= self.pano_caffee_text[scanId][c_new['viewpointId']][ix]['text_mask']
                c_new['obj_text']= self.pano_caffee_text[scanId][c_new['viewpointId']][ix]['text']
                new_relative_position = get_relative_position(loc_heading, base_heading, c_new['elevation'])
                c_new['obj_rel'] = new_relative_position
                candidate_new.append(c_new)
                # else:
                #     ix = c_new['pointId']
                #     normalized_heading = c_new['normalized_heading']
                #     visual_feat = feature[ix]
                #     loc_heading = normalized_heading - base_heading
                #     c_new['heading'] = loc_heading
                #     angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                #     c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                #     c_new['obj_feat']= self.pano_caffee[scanId][viewpointId][ix]['text_feature']
                #     c_new['obj_mask']= self.pano_caffee[scanId][viewpointId][ix]['text_mask']
                #     new_relative_position = get_relative_position(loc_heading, base_heading, c_new['elevation'])
                #     c_new['obj_rel'] = new_relative_position
                #     candidate_new.append(c_new)
                    
            return candidate_new
            
    def get_pano_obj(self, scan_id, viewpoint_id):
        pano_obj_feat = np.zeros((args.views, 36, 2048), dtype=np.float32)
        pano_obj_mask = np.zeros((args.views, 36), dtype=np.float32)
        pano_obj_boxes =  np.zeros((args.views, 36, 4), dtype=np.float32)
        pano_obj_text = []
        # for key, value in self.pano_caffee[scan_id][viewpoint_id].items():
        #     pano_obj_feat[key,:] = value
        #     pano_obj_mask[key,:] = value['text_mask']
        return pano_obj_feat, pano_obj_mask, pano_obj_text, pano_obj_boxes
  

    def _get_obs(self):
        obs = []
        
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
        
            pano_obj_feat, pano_obj_mask, pano_obj_text, pano_obj_boxes = self.get_pano_obj(state.scanId, state.location.viewpointId)
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
          
            # if args.test_obj:
            #     print("ERROR")
            # else:
            #     cand_obj_list = self.object_feats(self.pano_caffee, candidate, top_N_obj, state.location.viewpointId)
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                #'configurations': item['configurations'],
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id'],
                'pano_obj_feat': pano_obj_feat,
                'pano_obj_mask': pano_obj_mask,
                'pano_obj_text': pano_obj_text,
                'pano_obj_boxes': pano_obj_boxes
            })
            if 'configurations' in item:
                obs[-1]['configurations'] = item['configurations']
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats