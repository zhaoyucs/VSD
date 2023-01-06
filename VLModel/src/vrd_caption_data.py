from re import sub
from dataclasses import replace
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
vrd_dir = dataset_dir.joinpath('sp3000')
vg_dir = dataset_dir.joinpath('vg')
vg_feature_dir = vg_dir.joinpath('features')
vrd_img_dir = vrd_dir.joinpath('images/')
vrd_feature_dir = vrd_dir.joinpath('features')

predicate = ["on", "to the left of", "under", "behind", "to the right of", "in", "next to", "in front of", "above"]
predicate_map = {p: i for i, p in enumerate(predicate)}

class VRDCaptionFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vrd_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        # data_info_path = dataset_dir.joinpath(f'sp3000/{split}.json')
        data_info_path = dataset_dir.joinpath(f'spall/{split}.json')

        with open(data_info_path) as f:
            dataset = json.load(f)

        n_images = 0

        data = []
        for datum in dataset:
            img_id = datum['img_id'].replace('.jpg', "")
            img_id = img_id.replace('.png', "")
            if self.mode == 'train':
                for d in datum['captions']:
                    new_datum = {
                        'img_id': img_id,
                        'sent': d.strip(),
                        'subject_and_objects': [[triple['s'], triple['p'], triple['o']] for triple in datum['triple_list']],
                        "predicate": [triple['p'] for triple in datum['triple_list']],
                        'targets': [caption.strip() for caption in datum['captions']],
                        "so_bbox": [[triple['s_bbox'], triple['o_bbox']] for triple in datum['triple_list']],
                        'is_train': True
                    }
                    data.append(new_datum)
            else:
                new_datum = {
                    'img_id': img_id,
                    # 'subject_and_objects': [(triple['s'], triple['o']) for triple in datum['triple_list']],
                    'subject_and_objects': [[triple['s'], triple['p'], triple['o']] for triple in datum['triple_list']],
                    'targets': [caption.strip() for caption in datum['captions']],
                    "so_bbox": [[triple['s_bbox'], triple['o_bbox']] for triple in datum['triple_list']],
                    'is_train': False
                }
                data.append(new_datum)
                n_images += 1
                
        if self.verbose:
            print(f"{self.source} has f'{n_images}' images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        if self.args.max_n_boxes == 36:
            self.source_to_h5 = vrd_dir.joinpath('features').joinpath('vrd_boxes36.h5')

    def bbox_embed(self, bbox_s, bbox_o, w, h):
        ## spatialsense
        ##[miny, maxy, minx, maxx] left-up to right-down, 0 point is left-up
        ys,ys2,xs,xs2 = bbox_s
        yo,yo2,xo,xo2 = bbox_o

        ws = abs(xs - xs2)
        hs = abs(ys - ys2)
        wo = abs(xo - xo2)
        ho = abs(yo - yo2)

        sr1 = 1.0 * xs / ws
        sr2 = 1.0 * ys / hs
        sr3 = 1.0 * xs2 / ws
        sr4 = 1.0 * ys2 / hs

        or1 = 1.0 * xo / wo
        or2 = 1.0 * yo / ho
        or3 = 1.0 * xo2 / wo
        or4 = 1.0 * yo2 / ho

        # sr1 = 1.0 * xs / w
        # sr2 = 1.0 * ys / h
        # sr3 = 1.0 * xs2 / w
        # sr4 = 1.0 * ys2 / h

        # or1 = 1.0 * xo / w
        # or2 = 1.0 * yo / h
        # or3 = 1.0 * xo2 / w
        # or4 = 1.0 * yo2 / h

        
        return torch.tensor([[sr1, sr2, sr3, sr4],[or1, or2, or3, or4]])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            f = self.source_to_h5

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5 = f

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            if self.args.n_boxes == 100:
                assert n_boxes == 100
                assert len(feats) == 100
                assert len(boxes) == 100

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

        
        ##### so bboex #####
        raw_bbox = [self.bbox_embed(e[0], e[1], img_w, img_h) for e in datum['so_bbox']]

        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []

        else:
            if self.args.prefix is None:
                prefix = 'caption:'
            elif self.args.prefix == 'span':
                prefix = "span prediction:"
            elif self.args.prefix == 'denoise':
                prefix = "denoise text: <mask>"
            elif self.args.prefix == 'mask':
                if 'bart' in self.args.tokenizer:
                    prefix = "<mask>"

            prefix = 'describe image with tags and relation:'
            # prefix_vrd = 'relation detection:'
            prefix_vrd = prefix
            input_tokens = [prefix_vrd]
            input_tokens_with_vrd = [prefix]

            # if self.args.oscar_tags:
            #     prefix = 'describe image with tags:'
            #     input_tokens = [prefix]
            #     obj_ids = f[f'{img_id}/obj_id'][()]
            #     for obj_id in obj_ids:
            #         obj = self.vg_classes[obj_id]
            #         if obj not in input_tokens:
            #             input_tokens.append(obj)

            # Generate Visual Relation Caption
            # [Prefix] + [subject_0] + [extra_id_0] + [object_0] + [subject_1] + [extra_id_1] + [object_1]  
            # [extra_id_{i}] is used for relation classification
            predciate_sequence = []
            for i, subject_and_object in enumerate(datum['subject_and_objects']):
                if self.args.use_gold_rels:
                    input_tokens.extend(subject_and_object)
                else:
                    input_tokens.append(subject_and_object[0])
                    # input_tokens.append(f'<extra_id_{i}>')
                    input_tokens.append(f'<extra_id_0>')
                    
                    # input_tokens.append('<extra_id_%d>' % (predicate_map[subject_and_object[1]] + 1))
                    ### random replace
                    # vrd_token = '<extra_id_%d>' % (predicate_map[subject_and_object[1]] + 1)
                    # if random.random() <= 0.9:
                    #     from random import randint
                    #     i = randint(0, 9)
                    #     vrd_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.convert_tokens_to_ids('<extra_id_1>')-i)
                    # input_tokens.append(vrd_token)
                    ###
                    predciate_sequence.append(subject_and_object[1])
                    input_tokens.append(subject_and_object[2])

                    input_tokens_with_vrd.append(subject_and_object[0])
                    input_tokens_with_vrd.append('<extra_id_%d>' % (predicate_map[subject_and_object[1]] + 1))
                    input_tokens_with_vrd.append(subject_and_object[2])


            input_text = ' '.join(input_tokens)
            input_text_with_vrd = ' '.join(input_tokens_with_vrd)

            if 't5' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
                input_ids_with_vrd = self.tokenizer.encode(
                    input_text_with_vrd,
                    max_length=self.args.max_text_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
                input_ids_with_vrd = self.tokenizer.encode(
                    input_text_with_vrd,
                    max_length=self.args.max_text_length, truncation=True)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])
            
        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['input_length_vrd'] = len(input_ids_with_vrd)

        out_dict['input_ids_with_vrd'] = torch.LongTensor(input_ids_with_vrd)
        
        if datum['is_train']:
            sent = datum['sent'].strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            
            so_bbox = []
            zero_bbox = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0]])
            if not self.args.use_gold_rels:
                target_relation_ids = []
                predicate_start = 0
                for target_id in input_ids:
                    if target_id in self.tokenizer.additional_special_tokens_ids:
                        target_relation_ids.append(predicate_map[predciate_sequence[predicate_start]])
                        so_bbox.append(raw_bbox[predicate_start])
                        predicate_start += 1
                    else:
                        so_bbox.append(zero_bbox)
                        target_relation_ids.append(-1)
                out_dict['target_relation_ids'] = torch.LongTensor(target_relation_ids)
                out_dict['target_relation_length'] = len(target_relation_ids)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            out_dict['sent'] = sent
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['so_bbox'] = torch.stack(so_bbox)
        else:
            so_bbox = []
            zero_bbox = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0]])
            target_relation_mask = []
            predicate_start = 0
            for target_id in input_ids:
                if target_id in self.tokenizer.additional_special_tokens_ids:
                    target_relation_mask.append(1)
                    so_bbox.append(raw_bbox[predicate_start])
                    predicate_start += 1
                else:
                    so_bbox.append(zero_bbox)
                    target_relation_mask.append(-1)
            out_dict['target_relation_ids'] = torch.LongTensor(target_relation_mask)
            out_dict['target_relation_length'] = len(target_relation_mask)

            out_dict['so_bbox'] = torch.stack(so_bbox)

        if 'targets' in datum:
            out_dict['targets'] = datum['targets']
            

       
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        S_W_L_V = max(entry['input_length_vrd'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_ids_with_vrd = torch.ones(B, S_W_L_V, dtype=torch.long) * self.tokenizer.pad_token_id
        so_bbox = torch.ones(B, S_W_L, 2, 4, dtype=torch.float32)

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)
            assert input_ids_with_vrd.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        
        if 'target_relation_ids' in batch[0]:
            TR_W_L = max(entry['target_relation_length'] for entry in batch)
            target_relation_ids = torch.ones(B, TR_W_L, dtype=torch.long) * -1
            assert TR_W_L == S_W_L
                
        # if 'so_bbox' in batch[0]:
        #     B_L = max(entry['so_bbox'] for entry in batch)
            

        sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_ids_with_vrd[i, :entry['input_length_vrd']] = entry['input_ids_with_vrd']
            so_bbox[i, :entry['input_length']] = entry['so_bbox']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']
            
            if 'target_relation_ids' in entry:
                target_relation_ids[i, :entry["target_relation_length"]] = entry['target_relation_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])
            


        batch_entry['input_ids'] = input_ids
        batch_entry['input_ids_with_vrd'] = input_ids_with_vrd
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        
        if 'target_relation_ids' in batch[0]:
            word_mask = target_relation_ids != -1
            target_relation_ids[~word_mask] = -100
            batch_entry['target_relation_ids'] = target_relation_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets

        batch_entry['so_bbox'] = so_bbox

        batch_entry['task'] = 'caption'

        return batch_entry

class VGRelationFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        data_info_path = dataset_dir.joinpath(f'vg/{split}.json')
        with open(data_info_path) as f:
            dataset = json.load(f)

        n_images = 0

        data = []
        for img_id in dataset:
            datum = dataset[img_id]
            img_id = img_id.replace('.jpg', "")
            img_id = img_id.replace('.png', "")
            if self.mode == 'train':
                new_datum = {
                    'img_id': img_id,
                    'subject_and_objects': [[triple['subject']['name'], triple['predicate'], triple['object']['name']] for triple in datum['relations']],
                    "predicate": [triple['predicate'] for triple in datum['relations']],
                    "so_bbox": [[[triple['subject']['x'], triple['subject']['y'], triple['subject']['w'], triple['subject']['h']], [triple['object']['x'], triple['object']['y'], triple['object']['w'], triple['object']['h']]] for triple in datum['relations']],
                    'is_train': True
                }
                data.append(new_datum)
            else:
                new_datum = {
                    'img_id': img_id,
                    'subject_and_objects': [[triple['subject']['name'], triple['predicate'], triple['object']['name']] for triple in datum['relations']],
                    "predicate": [triple['predicate'] for triple in datum['relations']],
                    "so_bbox": [[[triple['subject']['x'], triple['subject']['y'], triple['subject']['w'], triple['subject']['h']], [triple['object']['x'], triple['object']['y'], triple['object']['w'], triple['object']['h']]] for triple in datum['relations']],
                    'is_train': False
                }
                data.append(new_datum)
                n_images += 1
                
        if self.verbose:
            print(f"{self.source} has f'{n_images}' images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        if self.args.max_n_boxes == 36:
            self.source_to_h5 = vg_dir.joinpath('features').joinpath('vg_gqa_obj36.h5')
    
    def bbox_embed_vg(self, bbox_s, bbox_o, w, h):
        ##[miny, maxy, minx, maxx] left-up to right-down, 0 point is left-up
        # ys,ys2,xs,xs2 = bbox_s
        # yo,yo2,xo,xo2 = bbox_o

        # ws = abs(xs - xs2)
        # hs = abs(ys - ys2)
        # wo = abs(xo - xo2)
        # ho = abs(yo - yo2)

        ## VG
        ## x,y the left-top corner
        xs,ys,ws,hs = bbox_s
        xo,yo,wo,ho = bbox_o

        xs2 = xs + ws
        ys2 = ys + hs
        xo2 = xo + wo
        yo2 = yo + ho
        # sr1 = 1.0 * xs / ws
        # sr2 = 1.0 * ys / hs
        # sr3 = 1.0 * xs2 / ws
        # sr4 = 1.0 * ys2 / hs

        # or1 = 1.0 * xo / wo
        # or2 = 1.0 * yo / ho
        # or3 = 1.0 * xo2 / wo
        # or4 = 1.0 * yo2 / ho
        sr1 = 1.0 * xs / w
        sr2 = 1.0 * ys / h
        sr3 = 1.0 * xs2 / w
        sr4 = 1.0 * ys2 / h

        or1 = 1.0 * xo / w
        or2 = 1.0 * yo / h
        or3 = 1.0 * xo2 / w
        or4 = 1.0 * yo2 / h

        return torch.tensor([[sr1, sr2, sr3, sr4],[or1, or2, or3, or4]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]

        ###### Image ######
        if self.args.use_vision:
            img_id = datum['img_id']
            out_dict['img_id'] = img_id

            f = self.source_to_h5

            if isinstance(f, Path):
                # path = self.data_source_to_h5_path[source]
                f = h5py.File(f, 'r')
                # self.split_to_h5_features[split_i] = f
                self.source_to_h5 = f

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)

            boxes.clamp_(min=0.0, max=1.0)

            n_boxes = len(boxes)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            feats = torch.from_numpy(feats)

            if self.args.n_boxes == 100:
                assert n_boxes == 100
                assert len(feats) == 100
                assert len(boxes) == 100

            n_boxes = min(n_boxes, self.args.max_n_boxes)
            out_dict['n_boxes'] = n_boxes
            boxes = boxes[:n_boxes]
            feats = feats[:n_boxes]
            out_dict['boxes'] = boxes
            out_dict['vis_feats'] = feats

        
        ##### so bboex #####
        raw_bbox = [self.bbox_embed_vg(e[0], e[1], img_w, img_h) for e in datum['so_bbox']]

        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []

        else:
            if self.args.prefix is None:
                prefix = 'caption:'
            elif self.args.prefix == 'span':
                prefix = "span prediction:"
            elif self.args.prefix == 'denoise':
                prefix = "denoise text: <mask>"
            elif self.args.prefix == 'mask':
                if 'bart' in self.args.tokenizer:
                    prefix = "<mask>"

            prefix = 'describe image with tags and relation:'
            input_tokens = [prefix]
            input_tokens_with_vrd = [prefix]

            # if self.args.oscar_tags:
            #     prefix = 'describe image with tags:'
            #     input_tokens = [prefix]
            #     obj_ids = f[f'{img_id}/obj_id'][()]
            #     for obj_id in obj_ids:
            #         obj = self.vg_classes[obj_id]
            #         if obj not in input_tokens:
            #             input_tokens.append(obj)

            # Generate Visual Relation Caption
            # [Prefix] + [subject_0] + [extra_id_0] + [object_0] + [subject_1] + [extra_id_1] + [object_1]  
            # [extra_id_{i}] is used for relation classification
            predciate_sequence = []
            for i, subject_and_object in enumerate(datum['subject_and_objects']):
                if self.args.use_gold_rels:
                    input_tokens.extend(subject_and_object)
                else:
                    input_tokens.append(subject_and_object[0])
                    # input_tokens.append(f'<extra_id_{i}>')
                    input_tokens.append(f'<extra_id_0>')
                    predciate_sequence.append(subject_and_object[1])
                    input_tokens.append(subject_and_object[2])

                    input_tokens_with_vrd.append(subject_and_object[0])
                    input_tokens_with_vrd.append('<extra_id_%d>' % (predicate_map[subject_and_object[1]] + 1))
                    input_tokens_with_vrd.append(subject_and_object[2])


            input_text = ' '.join(input_tokens)
            input_text_with_vrd = ' '.join(input_tokens_with_vrd)

            if 't5' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
                input_ids_with_vrd = self.tokenizer.encode(
                    input_text_with_vrd,
                    max_length=self.args.max_text_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
                input_ids_with_vrd = self.tokenizer.encode(
                    input_text_with_vrd,
                    max_length=self.args.max_text_length, truncation=True)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])
            
        out_dict['input_text'] = input_text

        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        out_dict['input_ids_with_vrd'] = torch.LongTensor(input_ids_with_vrd)
        
        if datum['is_train']:
            # sent = datum['sent'].strip()
            # if 't5' in self.args.tokenizer:
            #     target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            # elif 'bart' in self.args.tokenizer:
            #     target_ids = self.tokenizer.encode(sent, max_length=self.args.gen_max_length, truncation=True)
            
            so_bbox = []
            zero_bbox = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0]])
            if not self.args.use_gold_rels:
                target_relation_ids = []
                predicate_start = 0
                for target_id in input_ids:
                    if target_id in self.tokenizer.additional_special_tokens_ids:
                        target_relation_ids.append(predicate_map[predciate_sequence[predicate_start]])
                        so_bbox.append(raw_bbox[predicate_start])
                        predicate_start += 1
                    else:
                        so_bbox.append(zero_bbox)
                        target_relation_ids.append(-1)
                out_dict['target_relation_ids'] = torch.LongTensor(target_relation_ids)
                out_dict['target_relation_length'] = len(target_relation_ids)

            # assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            # out_dict['sent'] = sent
            # out_dict['target_ids'] = torch.LongTensor(target_ids)
            # out_dict['target_length'] = len(target_ids)

            out_dict['so_bbox'] = torch.stack(so_bbox)
        else:
            so_bbox = []
            zero_bbox = torch.tensor([[0, 0, 0, 0],[0, 0, 0, 0]])
            target_relation_mask = []
            predicate_start = 0
            for target_id in input_ids:
                if target_id in self.tokenizer.additional_special_tokens_ids:
                    target_relation_mask.append(1)
                    so_bbox.append(raw_bbox[predicate_start])
                    predicate_start += 1
                else:
                    so_bbox.append(zero_bbox)
                    target_relation_mask.append(-1)
            out_dict['target_relation_ids'] = torch.LongTensor(target_relation_mask)
            out_dict['target_relation_length'] = len(target_relation_mask)

            out_dict['so_bbox'] = torch.stack(so_bbox)

        if 'targets' in datum:
            out_dict['targets'] = datum['targets']
            

       
        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        input_ids_with_vrd = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        so_bbox = torch.ones(B, S_W_L, 2, 4, dtype=torch.float32)

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)
            assert input_ids_with_vrd.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        
        if 'target_relation_ids' in batch[0]:
            TR_W_L = max(entry['target_relation_length'] for entry in batch)
            target_relation_ids = torch.ones(B, TR_W_L, dtype=torch.long) * -1
            assert TR_W_L == S_W_L
                
        # if 'so_bbox' in batch[0]:
        #     B_L = max(entry['so_bbox'] for entry in batch)
            

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            input_ids_with_vrd[i, :entry['input_length']] = entry['input_ids_with_vrd']
            so_bbox[i, :entry['input_length']] = entry['so_bbox']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_id'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']
            
            if 'target_relation_ids' in entry:
                target_relation_ids[i, :entry["target_relation_length"]] = entry['target_relation_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])
            


        batch_entry['input_ids'] = input_ids
        batch_entry['input_ids_with_vrd'] = input_ids_with_vrd
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids
        
        if 'target_relation_ids' in batch[0]:
            word_mask = target_relation_ids != -1
            target_relation_ids[~word_mask] = -100
            batch_entry['target_relation_ids'] = target_relation_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        # batch_entry['targets'] = targets

        batch_entry['so_bbox'] = so_bbox

        batch_entry['task'] = 'caption'

        return batch_entry



def get_loader(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = VRDCaptionFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
        # train_sampler = RandomNonreplacmentSampler(dataset, dataset.n_iter)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'vrd_caption'

    return loader


def get_loader_vg(args, split='train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    # if 'mscoco' in split:
    verbose = (gpu == 0)

    dataset = VGRelationFineTuneDataset(
        split,
        # raw_dataset=_dset,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)
    # elif 'CC' in split:
    #     dataset = CCDataset(split, transform=transform, topk=topk)

    if distributed and mode == 'train':
        # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        train_sampler = DistributedSampler(dataset)
        # train_sampler = RandomNonreplacmentSampler(dataset, dataset.n_iter)
    else:
        train_sampler = None
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    if verbose:
        loader.evaluator = COCOCaptionEvaluator()

    loader.task = 'vrd_caption'

    return loader



class COCOCaptionEvaluator:
    def __init__(self):
        import language_evaluation
        self.evaluator = language_evaluation.CocoEvaluator(verbose=False)


    def evaluate(self, predicts, answers):

        results = self.evaluator.run_evaluation(predicts, answers)

        return results