"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from typing import Optional, List
from typing import Any, Dict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from torch import Tensor
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans

from lavis.models.Qwen_models.Qwen import Qwen
import random
Kwargs = Dict[str, Any]

def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return 
def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return 
def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
    x = torch.matmul(query,target).squeeze().to(query.device)
    #print('x',x.shape)
    sim_i2t,_ = x.max(-1)
    sim_i2t = sim_i2t / 0.07
    return F.cross_entropy(sim_i2t, targets)

class Abstractvilu(nn.Module):
    def __init__(
        self,
        fusion_dim: int = 256,
        target_dim: int = 256,
        layers: List[int] = [256,128,1], # [256,256,128,1] [256,128,64,1] [256,128,32,1] [256,128,3,1]
        activation: str = "relu",

        negative_slope: float = 0.01,
        use_sigmoid: bool = True,
        logit_scale: float = 100.0,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.target_dim = target_dim
        self.layers = layers
        self.activation = activation
        self.negative_slope = negative_slope
        self.use_sigmoid = use_sigmoid
        self.logit_scale = logit_scale
        self.activation_fn = self.get_activation()

    def forward(
        self,
        v: Tensor,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def build_mlp(self, input_dim: int, rate) -> nn.Module:
        modules = [nn.Linear(input_dim, self.layers[0]), self.activation_fn]
        # 在第一个线性层和激活函数后加入Dropout
        modules.append(nn.Dropout(p=rate))

        for i in range(len(self.layers) - 1):
            modules.append(nn.Linear(self.layers[i], self.layers[i + 1]))
        
            # 如果不是最后一个线性层，则添加激活函数和Dropout
            if i < len(self.layers) - 2:
                modules.append(self.activation_fn)
                modules.append(nn.Dropout(p=rate))

        mlp = nn.Sequential(*modules)
        return mlp

    def get_activation(self) -> nn.Module:
        if self.activation == "relu":
            activation_fn = nn.ReLU()
        elif self.activation == "leaky_relu":
            activation_fn = nn.LeakyReLU(self.negative_slope)
        else:
            raise ValueError("Activation function not supported")

        return activation_fn


class viluChange(Abstractvilu):
    def __init__(
        self,
        concat: bool = False,
        alpha:float = 0.5,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.concat = concat
        self.alpha=alpha
        if not self.concat:
            assert self.fusion_dim == self.target_dim
        self.input_dim = self.fusion_dim * 4
        self.mlp = None
    def forward(
        self,
        v: Tensor,
        t: Tensor,
    ) -> Tensor:
        
        fusion_feats=v
        target_feats=t

        fusion_feats = fusion_feats.squeeze()
        target_feats = target_feats.mean(-1)
         
        
        x = torch.concat((fusion_feats,target_feats,fusion_feats-target_feats,fusion_feats * target_feats),dim=-1)
        for layer in self.mlp:
            x = layer(x)
        # alpha ： 不确定性
        
        w = x.sigmoid()

        mask = torch.matmul(w,torch.ones(1,w.shape[0]).to(w.device))

        mask.to(w.device)

        return mask , x


@registry.register_model("Blip2QformerCir")
class Blip2QformerCir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        alpha=0.5,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)

        #vilu
        # self.vilu = viluAttention(
        #     keep_frozen=False,
        #     concat=True,
        #     identity_init=True,
        #     alpha=alpha, 
        # )
        self.vilu = viluChange(
            concat=True,
            alpha=alpha,
        )
        # self.vilu.load_state_dict(torch.load(vilu_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')['state_dict'])
        self.vilu.to(self.device)
        # self.vilu.eval()
    def setdrop(self,drop):
        self.vilu.mlp = self.vilu.build_mlp(self.vilu.input_dim,drop)
    def read_digits_to_tensor(self,file_path,step):
        digits_list = []
        with open(file_path, 'r') as file:
            # 逐行读取文件
            for i,line in enumerate(file):
                # 移除行末换行符和空格
                if(i != step):
                    continue
                cleaned_line = line.strip()
                # 遍历当前行的每个字符
                for char in cleaned_line:
                    # 确保字符是数字
                    if char.isdigit():
                        digits_list.append(int(char))
        # 将列表转换为Tensor
        if len(digits_list) != 256 :
            digits_list = digits_list[:256]
        return torch.tensor(digits_list)

    
    def loss_recon(self,query,target,c,alpha):
        
        #做一个相似度计算
        x = torch.matmul(query,target).squeeze().to(query.device)
        # 256，256，32 -> 256，256
        sim_i2t,_ = x.max(-1)
        
        sim_i2t =( sim_i2t / 0.07 )
        
        # 取对角线把维度变成 256，256
        print('sim_i2t',sim_i2t)
        print('sim_i2t_m',sim_i2t.mean())
        # max(sim(z,q)-δ，0)
        sim_i2t = torch.max(sim_i2t - alpha, torch.tensor(0.).to(sim_i2t.device))
        # shape 256，1，将噪声的分数提高
        c = c.sigmoid()
        c = 1-c
        c_ones = torch.ones((1,c.shape[0])).to(c.device)

        c = torch.matmul(c,c_ones)
        return (sim_i2t * c).sum() / (c.sum() / c.shape[0])
    
    def loss_bce(
        self,
        input: Tensor,
        labels: Tensor,
        reduction: str = "mean",
    ):

        #通过 labels 和 correct 比对得到 一个最终的 target
        labels = labels.to(self.device)
        correct = labels 
        # 加权
        target = correct.float().to(self.device)

        rate = (target == 0).sum() / (target == 1).sum()
        
        weights = torch.ones_like(target)
        weights[target == 1] = rate

        # weights[correct] *= torch.log(1 + (acc / (1 - acc)))

        # 不加权
        # weights[correct] *= weight
        loss = F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)
        
        return loss

    def robust_infoNCE(self, query, target, ifViLUBatch):
    
        with torch.no_grad() if ifViLUBatch else torch.enable_grad():
            vilu_mask , confid_scores  = self.vilu(query, target)

        eps=1e-7
        bs = query.size(0)
        x = torch.matmul(query,target).squeeze().to(query.device)
        sim_i2t,_ = x.max(-1)
        i2t=(sim_i2t/ 0.07).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        labels = torch.arange(query.shape[0]).long().cuda()
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), labels] = 0.

        if ifViLUBatch:
            loss = - ((1. - i2t).log() * mask * vilu_mask).sum() / bs
        else:
            loss = - ((1. - i2t).log() * mask).sum() / bs

        return loss , confid_scores

    def forward(self,samples,device):
        image = samples["image"]
        target = samples["target"]
        text = samples["text_input"]
        
        vilu = samples["vilu"]
        warm = samples['warmup']
        ###============== reference text fusion ===================###
        # reference image feature  
        with torch.no_grad() if vilu else torch.enable_grad():

            # print("梯度",not vilu)

            image_embeds = self.ln_vision(self.visual_encoder(image))
            #torch.Size([8, 257, 1408])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            # query tokens
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )
            # text tokens
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            # fusion reference image and text tokens into a set of multi-modal tokens
            attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
            fusion_output = self.Qformer.bert(
                text_tokens.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            taregt_embeds = self.ln_vision(self.visual_encoder(target))
            target_atts = torch.ones(taregt_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            target_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=taregt_embeds,
                encoder_attention_mask=target_atts,
                use_cache=True,
                return_dict=True,
            )
            #Target fea
            target_feats = F.normalize(
                self.vision_proj(target_output.last_hidden_state), dim=-1
            )

            #fusion fea
            fusion_feats = F.normalize(
                self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
            )
            fusion_feats=fusion_feats.unsqueeze(1).unsqueeze(1)
            target_feats=target_feats.permute(0, 2, 1)
            # loss_stu_rank = info_nce(fusion_feats, target_feats)#Eq13
            
        if(vilu):
            labels = samples["labels"]
            vilu_mask , confid_scores = self.vilu(fusion_feats, target_feats)
            loss_vilu = self.loss_bce(confid_scores.squeeze(-1),labels)
            loss = {
                'loss_vilu':loss_vilu,
                }
        else:
            loss_stu_rank,confid_scores =self.robust_infoNCE(fusion_feats,target_feats,vilu)
            loss_recon = torch.tensor(0.0, device=device, requires_grad=False) if warm else self.loss_recon(fusion_feats,target_feats,confid_scores,0.5)
            loss = {
                'loss_stu_rank':loss_stu_rank,
                'loss_recon':loss_recon
            }

        return loss
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text, return_attns=False):
        reference_embeds = reference_embeds.cuda()
        target_feats = target_feats.cuda()
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / 0.07

        if return_attns:
            return sim_i2t, fusion_output.cross_attentions[6].mean(1)

        return sim_i2t
    
    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
        #     attention_mask=attention_mask,
        #     return_dict=True,
        # )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen


    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
