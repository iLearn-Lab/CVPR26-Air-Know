"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import string 
import glob
import torchvision
import random
import pickle
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import trange
import logging
import base64
import time
def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, transform=None,noise_ratio=0):
        super().__init__()
        self.split = 'original-split'
        self.noise_ratio = noise_ratio
        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json')):
            self.fashioniq_data = []
            self.train_init_process()
            with open(os.path.join(self.path, 'fashion_iq_data.json'), 'w') as f:
                json.dump(self.fashioniq_data, f)

            self.test_queries_dress, self.test_targets_dress = self.get_test_data('dress')
            self.test_queries_shirt, self.test_targets_shirt = self.get_test_data('shirt')
            self.test_queries_toptee, self.test_targets_toptee = self.get_test_data('toptee')
            save_obj(self.test_queries_dress, os.path.join(self.path, 'test_queries_dress.pkl'))
            save_obj(self.test_targets_dress, os.path.join(self.path, 'test_targets_dress.pkl'))
            save_obj(self.test_queries_shirt, os.path.join(self.path, 'test_queries_shirt.pkl'))
            save_obj(self.test_targets_shirt, os.path.join(self.path, 'test_targets_shirt.pkl'))
            save_obj(self.test_queries_toptee, os.path.join(self.path, 'test_queries_toptee.pkl'))
            save_obj(self.test_targets_toptee, os.path.join(self.path, 'test_targets_toptee.pkl'))

        else:
            with open(os.path.join(self.path, 'fashion_iq_data.json'), 'r') as f:
                self.fashioniq_data = json.load(f) 
            self.test_queries_dress = load_obj(os.path.join(self.path, 'test_queries_dress.pkl'))
            self.test_targets_dress = load_obj(os.path.join(self.path, 'test_targets_dress.pkl'))
            self.test_queries_shirt = load_obj(os.path.join(self.path, 'test_queries_shirt.pkl'))
            self.test_targets_shirt = load_obj(os.path.join(self.path, 'test_targets_shirt.pkl'))
            self.test_queries_toptee = load_obj(os.path.join(self.path, 'test_queries_toptee.pkl'))
            self.test_targets_toptee = load_obj(os.path.join(self.path, 'test_targets_toptee.pkl'))
        self.labels = [1] * len(self.fashioniq_data)
        self.qlabels = [1] * len(self.fashioniq_data)
        print(f"FashionIQ dataset initialized with {len(self.fashioniq_data)} samples.")
    def shuffle(self):
        logging.info(f'Shuffle data with noise_ratio {self.noise_ratio}.')
        if self.noise_ratio == 0:
            logging.info("无噪声模式")
            return

        num_samples = len(self.fashioniq_data)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]

        noise_candidate = [self.fashioniq_data[i]['candidate'] for i in shuffle_candidate_indices]
        noise_captions = [self.fashioniq_data[i]['captions'] for i in shuffle_captions_indices]
        noise_target = [self.fashioniq_data[i]['target'] for i in shuffle_target_indices]

        shuffle_indices.sort()
        for i in shuffle_indices:
            self.labels[i] = 0
            self.qlabels[i] = 0
        
        shuffle_indices_ = random.sample(range(num_samples), int(0.2 * num_samples))
        for i in shuffle_indices_:
            self.qlabels[i] = 1 - self.qlabels[i]
        
        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)

        for i in shuffle_candidate_indices:
            self.fashioniq_data[i]['candidate'] = noise_candidate.pop()
        for i in shuffle_captions_indices:
            self.fashioniq_data[i]['captions'] = noise_captions.pop()
        for i in shuffle_target_indices:
            self.fashioniq_data[i]['target'] = noise_target.pop()

        logging.info('Shuffle done.')

    def train_init_process(self):
        for name in ['dress', 'shirt', 'toptee']:
            with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(name, 'train')), 'r') as f:
                ref_captions = json.load(f)
            with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
                correction_dict = json.load(f)
            for triplets in ref_captions:
                ref_id = triplets['candidate']
                tag_id = triplets['target']
                cap = self.concat_text(triplets['captions'], correction_dict)
                self.fashioniq_data.append({
                    'target': name + '_' + tag_id,
                    'candidate': name + '_' + ref_id,
                    'captions': cap
                })

    def correct_text(self, text, correction_dict):
        trans=str.maketrans({key: ' ' for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])

        return text

    def concat_text(self, captions, correction_dict):
        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
        return text

    def __len__(self):
        return len(self.fashioniq_data)

    def __getitem__(self, idx):
        caption = self.fashioniq_data[idx]
        # mod_str = self.concat_text(caption['captions'])
        mod_str = caption['captions']
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        out['reference'] = caption['candidate']
        out['target_hard'] = caption['target']
        out['r_img_path'] = os.path.join(self.image_dir, candidate.split('_')[0], candidate.split('_')[1] + ".jpg")
        out['t_img_path'] = os.path.join(self.image_dir, target.split('_')[0], target.split('_')[1] + ".jpg")
        out['labels'] = self.labels[idx]
        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img

    def get_test_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
        img = self.transform(img)

        return img

    def get_all_texts(self):
        texts = []
        for caption in self.fashioniq_data:
            mod_texts = caption['captions']
            texts.append(mod_texts)
        return texts
    def get_test_data(self, name):       # query

        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:
            images = json.load(f)
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(name, 'val')), 'r') as f:
            ref_captions = json.load(f)
        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
            correction_dict = json.load(f)
        test_queries = []
        for idx in range(len(ref_captions)):
            caption = ref_captions[idx]
            mod_str = self.concat_text(caption['captions'], correction_dict)
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = images.index(candidate)
            out['source_img_data'] = self.get_test_img(name + '_' + candidate)
            out['target_img_id'] = images.index(target)
            out['target_img_data'] = self.get_test_img(name + '_' + target)
            out['mod'] = {'str': mod_str}

            test_queries.append(out)
        
        test_targets_id = []
        test_targets = []
        if self.split == 'val-split':
            for i in test_queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])

            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(name + '_' + images[i])
                test_targets.append(out)
        elif self.split == 'original-split':
            for id, image_name in enumerate(images):
                test_targets_id.append(id)
                out = {}
                out['target_img_id'] = id
                out['target_img_data'] = self.get_img(name + '_' + image_name)      
                test_targets.append(out)
        return test_queries, test_targets

class FashionIQ_V(torch.utils.data.Dataset):
    def __init__(self, path, transform=None,noise_ratio=0):
        super().__init__()
        self.split = 'original-split'
        self.noise_ratio = noise_ratio
        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.transform = transform
        
        with open(os.path.join(self.path+'judge', f'judge.train.{self.noise_ratio}.json'), 'r') as f:
                self.fashioniq_data = json.load(f) 
        self.test_queries_dress = load_obj(os.path.join(self.path, 'test_queries_dress.pkl'))
        self.test_targets_dress = load_obj(os.path.join(self.path, 'test_targets_dress.pkl'))
        self.test_queries_shirt = load_obj(os.path.join(self.path, 'test_queries_shirt.pkl'))
        self.test_targets_shirt = load_obj(os.path.join(self.path, 'test_targets_shirt.pkl'))
        self.test_queries_toptee = load_obj(os.path.join(self.path, 'test_queries_toptee.pkl'))
        self.test_targets_toptee = load_obj(os.path.join(self.path, 'test_targets_toptee.pkl'))

    def __len__(self):
        return len(self.fashioniq_data)

    def __getitem__(self, idx):
        caption = self.fashioniq_data[idx]
        # mod_str = self.concat_text(caption['captions'])
        mod_str = caption['caption']
        candidate = caption['reference']
        target = caption['target_hard']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        out['Qlabels'] = caption['Qlabels']
        out['labels'] = caption['labels']
        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img


class CIRR(torch.utils.data.Dataset):
    def __init__(
                self,
                path,
                transform=None, 
                case_look=False,
                noise_ratio=0.2,
                v_batch = 0,
                ) -> None:
        super(CIRR, self).__init__()
        self.path = path
        self.caption_dir = self.path + 'captions'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform
        self.case_look = case_look
        self.noise_ratio=noise_ratio
        self.v_batch = v_batch
        # train data
        with open(os.path.join(self.caption_dir, "cap.rc2.train.json"), 'r') as f:
            self.cirr_data = json.load(f)
        self.labels = [1] * len(self.cirr_data)
        self.qlabels = [1] * len(self.cirr_data)
        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys())

        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            self.val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            self.val_image_path = json.load(f)
            self.val_image_name = list(self.val_image_path.keys())

        # val data
        if not os.path.exists(os.path.join(self.path, 'cirr_val_queries.pkl')):
            self.val_queries, self.val_targets = self.get_val_queries()
            save_obj(self.val_queries, os.path.join(self.path, 'cirr_val_queries.pkl'))
            save_obj(self.val_targets, os.path.join(self.path, 'cirr_val_targets.pkl'))
        else:
            self.val_queries = load_obj(os.path.join(self.path, 'cirr_val_queries.pkl'))
            self.val_targets = load_obj(os.path.join(self.path, 'cirr_val_targets.pkl'))
        # test data
        if not os.path.exists(os.path.join(self.path, 'cirr_test_queries.pkl')):
            self.test_name_list, self.test_img_data, self.test_queries = self.get_test_queries()
            save_obj(self.test_name_list, os.path.join(self.path, 'cirr_test_name_list.pkl'))
            save_obj(self.test_img_data, os.path.join(self.path, 'cirr_test_img_data.pkl'))
            save_obj(self.test_queries, os.path.join(self.path, 'cirr_test_queries.pkl'))
        else:
            self.test_name_list = load_obj(os.path.join(self.path, 'cirr_test_name_list.pkl'))
            self.test_img_data = load_obj(os.path.join(self.path, 'cirr_test_img_data.pkl'))
            self.test_queries = load_obj(os.path.join(self.path, 'cirr_test_queries.pkl'))

    def shuffle(self):
        logging.info(f'Shuffle data with noise_ratio {self.noise_ratio}.')
        if self.noise_ratio == 0:
            return
        num_samples = len(self.cirr_data)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))

        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]

        noise_candidate = [self.cirr_data[i]['reference'] for i in shuffle_candidate_indices]
        noise_captions = [self.cirr_data[i]['caption'] for i in shuffle_captions_indices]
        noise_target = [self.cirr_data[i]['target_hard'] for i in shuffle_target_indices]
        for i in shuffle_indices:
            self.labels[i] = 0
            self.qlabels[i] = 0

        shuffle_indices_ = random.sample(range(num_samples), int(0.2 * num_samples))

        for i in shuffle_indices_:
            self.qlabels[i] = 1 - self.qlabels[i]

        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)

        for i in shuffle_candidate_indices:
            temp = noise_candidate.pop()
            if self.cirr_data[i]['reference'] == temp:
                self.labels[i] = 1
            else:
                self.cirr_data[i]['reference'] = temp
        for i in shuffle_captions_indices:
            # self.cirr_data[i]['caption'] = noise_captions.pop()
            temp = noise_captions.pop()
            if self.cirr_data[i]['caption'] == temp:
                self.labels[i] = 1
            else:
                self.cirr_data[i]['caption'] = temp
        for i in shuffle_target_indices:
            # self.cirr_data[i]['target_hard'] = noise_target.pop()
            temp = noise_target.pop()
            if self.cirr_data[i]['target_hard'] == temp:
                self.labels[i] = 1
            else:
                self.cirr_data[i]['target_hard'] = temp

        logging.info('Shuffle done.')
        
    def __len__(self):
        return len(self.cirr_data)
    
    
    def __getitem__(self, idx):
        caption = self.cirr_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']

        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name])
        out['target_img_data'] = self.get_img(self.train_image_path[target_name])
        out['mod'] = {'str':mod_str}
        out['reference'] = reference_name
        out['target_hard'] = target_name
        out['r_img'] = os.path.join(self.path, self.train_image_path[reference_name].lstrip('./'))
        out['t_img'] = os.path.join(self.path, self.train_image_path[target_name].lstrip('./'))
        out['r_img_path'] = os.path.join(self.path, self.train_image_path[reference_name].lstrip('./'))
        out['t_img_path'] = os.path.join(self.path, self.train_image_path[target_name].lstrip('./'))
        out['labels'] = self.labels[idx]
        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = list(val_image_path.keys())
        
        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['caption']
            reference_name = caption['reference']
            target_name = caption['target_hard']
            subset_names = caption['img_set']['members']
            subset_ids = [val_image_name.index(n) for n in subset_names]

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_path[reference_name])
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_path[target_name])
            out['mod'] = {'str':mod_str}
            out['subset_id'] = subset_ids
            if self.case_look:
                out['raw_src_img_data'] = self.get_img(val_image_path[reference_name], return_raw=True)
                out['raw_tag_img_data'] = self.get_img(val_image_path[target_name], return_raw=True)
            
            test_queries.append(out)

        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_path[name])
            if self.case_look:
                out['raw_tag_img_data'] = self.get_img(val_image_path[name], return_raw=True)
            test_targets.append(out)

        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']])
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(test_image_path[name])
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries


class CIRR_V(torch.utils.data.Dataset):
    def __init__(
                self,
                path,
                transform=None, 
                case_look=False,
                noise_ratio=0.2,
                ) -> None:
        super(CIRR_V, self).__init__()
        self.path = path
        self.caption_dir = self.path + 'judge'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform
        self.case_look = case_look
        self.noise_ratio=noise_ratio
        # train data
        with open(os.path.join(self.caption_dir, f"judge.train.{self.noise_ratio}.json"), 'r') as f:
                self.vilu_data = json.load(f)
        with open(os.path.join(self.split_dir, "split.rc2.train.json"), 'r') as f:
            self.train_image_path = json.load(f)
            self.train_image_name = list(self.train_image_path.keys())

        # with open(os.path.join(self.caption_dir, "cap.rc2.val.json"), 'r') as f:
        #     self.val_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.val.json"), 'r') as f:
            self.val_image_path = json.load(f)
            self.val_image_name = list(self.val_image_path.keys())
        num_samples = len(self.vilu_data)
        # shuffle_indices_t = random.sample(range(num_samples), int(0.8* num_samples))

        # shuffle_indices_t.sort()
        # for i in shuffle_indices_t:
        #     self.vilu_data[i]['labels'] = 1 - self.vilu_data[i]['labels']
        
    def __len__(self):
        return len(self.vilu_data)

    def __getitem__(self, idx):
        caption = self.vilu_data[idx]
        reference_name = caption['reference']
        mod_str = caption['caption']
        target_name = caption['target_hard']
        
        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name])
        out['target_img_data'] = self.get_img(self.train_image_path[target_name])
        out['mod'] = {'str':mod_str}

        out['labels'] = caption['labels']
        out['Qlabels'] = caption['Qlabels']
        out['r_img_path'] = os.path.join(self.path, self.train_image_path[reference_name].lstrip('./'))
        out['t_img_path'] = os.path.join(self.path, self.train_image_path[target_name].lstrip('./'))

        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img




class LaSCo(torch.utils.data.Dataset):
    def __init__(self, path, spilt="train", transform=None, case_look=False) -> None:
        super(LaSCo, self).__init__()
        self.path = path
        self.caption_dir = self.path + 'captions'
        self.split_dir = self.path + 'image_splits'
        self.transform = transform
        self.case_look = case_look
        # train data
        if spilt == "train":
            with open(os.path.join(self.path, "lasco_train.json"), 'r') as f:
                self.lasco_data = json.load(f)

            with open(os.path.join(self.path, "lasco_train_corpus.json"), 'r') as f:
                self.train_image_path = json.load(f)
                self.train_image_name = list(self.train_image_path.keys())
        if spilt == "val":
            with open(os.path.join(self.path, "lasco_val.json"), 'r') as f:
                self.val_data = json.load(f)

            with open(os.path.join(self.path, "lasco_val_corpus.json"), 'r') as f:
                self.val_image_path = json.load(f)
                self.val_image_name = []#list(self.val_image_path.keys())
                self.val_image_map = {}
                for idx in range(len(self.val_image_path)):
                    caption = self.val_image_path[idx]
                    self.val_image_name.append(caption["id"])
                    self.val_image_map.update({str(caption["id"]):caption["path"]})
            print(len(self.val_image_name))
        # val data
        # if not os.path.exists(os.path.join(self.path, 'lasco_val_queries.pkl')):
        #     self.val_queries, self.val_targets = self.get_val_queries()
        #     save_obj(self.val_queries, os.path.join(self.path, 'lasco_val_queries.pkl'))
        #     save_obj(self.val_targets, os.path.join(self.path, 'lasco_val_targets.pkl'))
        # else:
        #     self.val_queries = load_obj(os.path.join(self.path, 'lasco_val_queries.pkl'))
        #     self.val_targets = load_obj(os.path.join(self.path, 'lasco_val_targets.pkl'))

    def __len__(self):
        return len(self.lasco_data)

    def __getitem__(self, idx):
        caption = self.lasco_data[idx]
        mod_str = caption['query-text']
        reference_name = str(caption['query-image'][0])
        target_name = str(caption['target-image'][0])
        
        out = {}
        out['source_img_data'] = self.get_img(self.train_image_path[reference_name])
        out['target_img_data'] = self.get_img(self.train_image_path[target_name])
        out['mod'] = {'str':mod_str}
        return out

    def get_img(self, img_path, return_raw=False):
        img_path = os.path.join(self.path, img_path.lstrip('./'))
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
            
        if return_raw:
            transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
            return transform(img)

        if self.transform:
            img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            #img = self.transform(img)
        return img

    def get_val_queries(self):
        with open(os.path.join(self.path, "lasco_val.json"), 'r') as f:
            val_data = json.load(f)

        with open(os.path.join(self.path, "lasco_val_corpus.json"), 'r') as f:
            val_image_path = json.load(f)
            val_image_name = []#list(val_image_path.keys())
            val_image_map = {}
            for idx in range(len(self.val_image_path)):
                caption = self.val_image_path[idx]
                val_image_name.append(str(caption["id"]))
                val_image_map.update({str(caption["id"]):caption["path"]})
        
        test_queries = []
        for idx in range(len(val_data)):
            caption = val_data[idx]
            mod_str = caption['query-text']
            reference_name = str(caption['query-image'][0])
            target_name = str(caption['target-image'][0])

            out = {}
            out['source_img_id'] = val_image_name.index(reference_name)
            out['source_img_data'] = self.get_img(val_image_map[reference_name])
            out['target_img_id'] = val_image_name.index(target_name)
            out['target_img_data'] = self.get_img(val_image_map[target_name])
            out['mod'] = {'str':mod_str}
            if self.case_look:
                out['raw_src_img_data'] = self.get_img(val_image_map[reference_name], return_raw=True)
                out['raw_tag_img_data'] = self.get_img(val_image_map[target_name], return_raw=True)
            
            test_queries.append(out)
        print(len(test_queries))
        test_targets = []
        for i in range(len(val_image_name)):
            name = val_image_name[i]
            out = {}
            out['target_img_id'] = i
            out['target_img_data'] = self.get_img(val_image_map[name])
            if self.case_look:
                out['raw_tag_img_data'] = self.get_img(val_image_map[name], return_raw=True)
            test_targets.append(out)
        print(len(test_targets))
        return test_queries, test_targets
    
    def get_test_queries(self):

        with open(os.path.join(self.caption_dir, "cap.rc2.test1.json"), 'r') as f:
            test_data = json.load(f)

        with open(os.path.join(self.split_dir, "split.rc2.test1.json"), 'r') as f:
            test_image_path = json.load(f)
            test_image_name = list(test_image_path.keys())

        queries = []
        for i in range(len(test_data)):
            out = {}
            caption = test_data[i]
            out['pairid'] = caption['pairid']
            out['reference_data'] = self.get_img(test_image_path[caption['reference']])
            out['reference_name'] = caption['reference']
            out['mod'] = caption['caption']
            out['subset'] = caption['img_set']['members']
            queries.append(out)

        image_name = []
        image_data = []
        for i in range(len(test_image_name)):
            name = test_image_name[i]
            data = self.get_img(test_image_path[name])
            image_name.append(name)
            image_data.append(data)
        return image_name, image_data, queries


class Birds(torch.utils.data.Dataset):
    def __init__(self, path, gallery_all=False, split = 'train',transform=None):
        super(Birds, self).__init__()

        self.path = path
        self.image_dir = self.path + 'bird'

        self.split = split
        self.transform = transform
        self.gallery_all = gallery_all

        self.test_targets = []
        self.test_queries = []

        if self.split == 'train':
            with open(os.path.join(self.path, "cap.{}.{}.json".format('birds', 'train')), 'r') as f:
                self.ref_captions = json.load(f)
            with open(os.path.join(self.path, "split.{}.{}.json".format('birds', 'train')), 'r') as f:
                self.images = json.load(f)
        else:
            with open(os.path.join(self.path, "cap.{}.{}-merge.json".format('birds', 'test')), 'r') as f:
                self.ref_captions = json.load(f)
            with open(os.path.join(self.path, "split.{}.{}-merge.json".format('birds', 'test')), 'r') as f:
                self.images = json.load(f)   
                
    
    def concat_text(self, captions):
        sentences = captions.split('.')
        split_animal1 = []
        for sentence in sentences:
            dot_se = sentence.split(',')
            for dot in dot_se:
                while_se = dot.split('while')
                for while_s in while_se:
                    if 'animal1' in while_s and 'animal2' not in while_s:
                        continue
                    split_animal1.append(while_s)

        text = '.'.join(split_animal1).replace("while", "").replace("both", "").replace("animal1's", "").replace("animal2's", "")
        
        token = text.split(" ")
        if len(token) > 60:
            token = token[-60:]
        return ' '.join(token)
    
    def __len__(self):
        return len(self.ref_captions)
        
    
    def __getitem__(self, idx):
        caption = self.ref_captions[idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}
        # if self.split == 'train':
        #     if caption in self.augment_captions:
        #         out['weight'] = torch.ones(1) * 0.8
        #     else:
        #         out['weight'] = torch.ones(1)

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir, image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            #img = self.transform(img)
        return img

    def get_all_texts(self):
        texts = []
        with open(os.path.join(self.path, "cap.{}.{}.json".format('birds', 'train')), 'r') as f:
            train_captions = json.load(f)
        for caption in train_captions:
            mod_texts = caption['captions']
            texts.append(mod_texts)
        return texts

    def get_test_queries(self):       # query
        self.test_queries = []
        for idx in range(len(self.ref_captions)):
            caption = self.ref_captions[idx]
            mod_str = self.concat_text(caption['captions'])
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = self.images.index(candidate)
            out['source_img_data'] = self.get_img(candidate)
            out['target_img_id'] = self.images.index(target)
            out['target_img_data'] = self.get_img(target)
            out['mod'] = {'str': mod_str}

            self.test_queries.append(out)
        
        return self.test_queries


    def get_test_targets(self):       # 所有的image
        if self.gallery_all:
            self.test_targets = []
            for idx in range(len(self.images)):
                target = self.images[idx]
                out = {}
                out['target_img_id'] = idx
                out['target_img_data'] = self.get_img(target)
                self.test_targets.append(out)
        else:
            test_targets_id = []
            queries = self.get_test_queries()
            for i in queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
        
            self.test_targets = []
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(self.images[i])
                self.test_targets.append(out)   
        return self.test_targets
    
    
class Fashion200k(torch.utils.data.Dataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []
        self.test_queries = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                                         '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ', filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i
            )

    def get_test_queries(self):
        return self.test_queries

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                #if not caption2id.has_key(c):
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                #if not parent2children_captions.has_key(p):
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)
   
    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}
        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            #img = self.transform(img, return_tensors="pt", data_format="channels_first")['pixel_values']
            img = self.transform(img)
        return img

