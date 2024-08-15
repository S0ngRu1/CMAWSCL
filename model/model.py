import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.2):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
       
        attention = torch.matmul(q, k.transpose(-2, -1))
        #print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        #print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)
        #print('attn_final.shape:{}'.format(attention.shape))

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        #print("query.shape:{}".format(query.shape))
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        #batch_size = key.size(0)
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        #print('key.shape:{}'.format(key.shape))
        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        attention = self.dot_product_attention(query, key, value, 
                                               scale, attn_mask)
        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        #print('attention_con_shape:{}'.format(attention.shape))
        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        #print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)
        return output


class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network 
    """
    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.2):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features 
    """
    def __init__(self, model_dim=768, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)
        
        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        
        self.fusion_linear = nn.Linear(model_dim*2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):

        output_1 = self.attention_1(image_output, text_output, text_output,
                                 attn_mask)
        
        output_2 = self.attention_2(text_output, image_output, image_output,
                                 attn_mask)
        
        
        #print('attention out_shape:{}'.format(output.shape))
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)
        
        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output

class MyModel(nn.Module):
    def __init__(self, args, model_dim = 768, num_layers=1, num_heads=8, ffn_dim=2048, dropout=0.3):
        super(MyModel, self).__init__()
        self.args = args
        self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
        self.image_classfier = Classifier(dropout, model_dim,2)
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
        self.text_classfier = Classifier(dropout, model_dim, 2)
        self.mm_classfier = Classifier(dropout, model_dim, 2)
        self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])


    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if self.args.method in ['CMAWSC']:
            text = self.text_encoder(text=text)
            image = torch.squeeze(image, 1)
            image = self.image_encoder(pixel_values=image)
            output_text = self.text_classfier(text[:, 0, :])
            output_image = self.image_classfier(image[:, 0, :])
            for fusion_layer in self.fusion_layers:
                fusion = fusion_layer( image[:, 0, :], text[:, 0, :])   
            output_mm = self.mm_classfier(fusion) 
            if infer:
                return output_mm
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(criterion(output_image, label))
            mmcLoss = self.WSConLoss(text[:, 0, :], fusion, output_text, output_mm, label)
            MMLoss_m = torch.mean(criterion(output_mm, label))
            MMLoss_sum = MMLoss_m + MMLoss_text + MMLoss_image + 0.1* mmcLoss
            return MMLoss_sum, MMLoss_m, output_mm
        
    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit

    def WSConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a = feature_a / feature_a.norm(dim=-1, keepdim=True)
        feature_b = feature_b / feature_b.norm(dim=-1, keepdim=True)
        if predict_a is not None:
            predict_a = torch.argmax(F.softmax(predict_a, dim=1), dim=1)
            predict_b = torch.argmax(F.softmax(predict_b, dim=1), dim=1)
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()
        a_pre = predict_a.eq(labels)  
        b_pre = predict_b.eq(labels)
        a_b_pre = torch.gt(a_pre | b_pre, 0)   
        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if a_pre[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)
        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] 
        for i in range(feature_b.shape[0]):
            if b_pre[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())
        return mean_log_pos


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.hidden_1 = LinearLayer(in_dim, 2048)      
        self.hidden_2 = LinearLayer(2048, 256)
        self.hidden_3 = LinearLayer(256, 64)
        self.classify = LinearLayer(64, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.hidden_1(input), inplace=False)
        input_d1 = self.post_dropout(input_p1)
        input_p2 = F.relu(self.hidden_2(input_d1), inplace=False)
        input_d2 = self.post_dropout(input_p2)
        input_p3 = F.relu(self.hidden_3(input_d2), inplace=False)
        output = self.classify(input_p3)
        return output






