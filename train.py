import os
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score , recall_score
from model.model import MyModel 
from data.dataloader import MMDataLoader, TextDataLoader, ImageDataLoader
from utils.metrics import collect_metrics
from utils.functions import save_checkpoint, load_checkpoint, dict_to_str


logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger('')


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def get_optimizer(model, args):
    text_enc_param = list(model.module.text_encoder.named_parameters())
    text_clf_param = list(model.module.text_classfier.parameters())
    img_enc_param = list(model.module.image_encoder.parameters())
    img_clf_param = list(model.module.image_classfier.parameters())  
    mm_clf_param = list(model.module.mm_classfier.parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {"params": [p for n, p in text_enc_param if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay_tfm, 'lr': args.lr_text_tfm},
            {"params": [p for n, p in text_enc_param if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
             'lr': args.lr_text_tfm},
            {"params": text_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_text_cls},
            {"params": img_enc_param, "weight_decay": args.weight_decay_tfm, 'lr': args.lr_img_tfm},
            {"params": img_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_img_cls},
            {"params": mm_clf_param, "weight_decay": args.weight_decay_other, 'lr': args.lr_mm_cls},
        ]
    optimizer = optim.Adam(optimizer_grouped_parameters)

    return optimizer



def valid(args, model, data=None, best_valid=None, nBetter=None, step=None):
    model.eval() 
    if best_valid is None:
        best_valid = 0.0
    with torch.no_grad():
        valid_loader = data
        y_pred = []
        y_true = []
        with tqdm(valid_loader, desc='Validation', unit='batch') as td:
            for batch in td:
                batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label = batch
                text = (text_input_ids.to(args.device), 
                        text_token_type_ids.to(args.device), 
                        text_attention_mask.to(args.device))
                image = batch_image.to(args.device)
                batch_label = batch_label.to(args.device)
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        te_true = torch.cat(y_true).data.cpu().numpy()
        te_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        cur_valid = accuracy_score(te_true, te_prob.argmax(1))
        recall = recall_score(te_true, te_prob.argmax(1))
        isBetter = cur_valid >= (best_valid + 1e-6)
        valid_results = {
        "accuracy": cur_valid,"recall": recall}
        valid_results.update(collect_metrics(args.dataset, te_true, te_prob))
        if isBetter:
            save_checkpoint(model, args.best_model_save_path)
            best_valid = cur_valid
            nBetter = 0
        else:
            nBetter += 1
    return valid_results, best_valid, nBetter
    

def train_valid(args, model, optimizer, scheduler=None, data=None):
    best_valid = 1e-5
    nBetter = 0
    total_step = 0
    gradient_accumulation_steps = 4
    for epoch in range(args.num_epoch):
        model.train()
        train_loader, valid_loader, test_loader = data
        y_pred = []
        y_true = []
        train_loss_m = 0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epoch}', unit='batch') as td:
            for batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = (text_input_ids.to(args.device), 
                        text_token_type_ids.to(args.device), 
                        text_attention_mask.to(args.device))
                image = batch_image.to(args.device)
                labels = batch_label.to(args.device).view(-1)
                loss, loss_m, logit_m = model(text, image, None, labels)
                loss = loss.sum()
                loss.backward()
                train_loss_m += loss_m.sum().item()
                y_pred.append(logit_m.cpu())
                y_true.append(batch_label.cpu())
                total_step += 1
                if total_step % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    total_step = 0  # Reset total_step after gradient update
            
        logits = torch.cat(y_pred)
        tr_true = torch.cat(y_true).data.cpu().numpy()
        tr_prob = F.softmax(logits, dim=1).data.cpu().numpy()
        train_accuracy = accuracy_score(tr_true, tr_prob.argmax(1))
        average_train_loss = train_loss_m / len(train_loader)
        logger.info(f'Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.4f}, Loss: {average_train_loss:.4f}')

        valid_results, best_valid, nBetter = valid(args, model, data=valid_loader, best_valid=best_valid, nBetter=nBetter)
        logger.info(f'Epoch {epoch + 1}, Validation Results: {valid_results}')
        if scheduler is not None:
            scheduler.step(train_accuracy)  
    return best_valid
    
    
def test_epoch(args,model, dataloader=None):
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        with tqdm(dataloader) as td:
            for batch_image, text_input_ids, text_token_type_ids, text_attention_mask, batch_label in td:
                text = text_input_ids.cuda(), text_token_type_ids.cuda(), text_attention_mask.cuda()
                image = batch_image.cuda()
                logit = model.module.infer(text, image, None)
                y_pred.append(logit.cpu())
                y_true.append(batch_label.cpu())
        logits = torch.cat(y_pred)
        true = torch.cat(y_true).data.cpu().numpy()
        prob = F.softmax(logits, dim=1).data.cpu().numpy()
    return prob, true

def train(args):
    train_loader, valid_loader, test_loader = MMDataLoader(args)
    data = train_loader, valid_loader, test_loader
    model = DataParallel(MyModel(args))
    model = model.to(args.device)
    
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    if not args.test_only:
        logger.info("Start training...")
        best_results = train_valid(args, model, optimizer, scheduler, data)

    load_checkpoint(model, args.best_model_save_path)

    te_prob, te_true = test_epoch(args,model, test_loader)
    best_results = collect_metrics(args.dataset, te_true, te_prob)
    logger.info("Test: " + dict_to_str(collect_metrics(args.dataset, te_true, te_prob)))

