import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from tools.utils import (
                    build_tokenizer,
                    build_clip_loss,
                    build_model_transform,
                    bulid_dataloader,
                    bulid_dataloader_dist
                    )
from tools.dist_tools import (
    is_master,
    is_parallel,
    de_parallel,
    torch_distributed_zero_first,
    setup_ddp_envs,
    RANK,
    WORLD_SIZE
)
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.nn as nn
from contextlib import contextmanager


from typing import Union, Tuple
from torch.optim.adamw import AdamW
from torch.optim import lr_scheduler
import numpy as np
import time
import pandas as pd
from torch import distributed as dist

def print_step(step_num, data, item='loss'):
    print("step: {:0>8d}{:>8s} {:s}: {:.4f}".format(step_num, '', item, data))

def save_checkpoint(model, step, save_dir):
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
        suffix = '%s-%s.pth' % (step, time.strftime("%Y-%H-%M-%S"))
        de_parallel(model).eval()
        torch.save(de_parallel(model).state_dict(), os.path.join(save_dir, suffix))
        de_parallel(model).train()

def save_metric(data:dict, save_csv):
    with open(save_csv, 'a') as file:
        if isinstance(data, dict):
            data = pd.DataFrame(data, index=[0])
        data.to_csv(file,
                    sep='\t',
                    index=False,
                    header=not os.stat(save_csv).st_size > 0)

def safe_state_dict(model: torch.nn.Module, state_dict):
    new_state_dict = model.state_dict()
    map_keys = []
    no_map_keys = []
    for k, v in new_state_dict.items():
        if k in state_dict.keys() and v.shape == state_dict[k].shape:
            new_state_dict[k] = state_dict[k]
            map_keys.append(k)
        else:
            no_map_keys.append(k)
    print('load %d/%d state keys' % (len(map_keys), len(new_state_dict)))
    return new_state_dict

        
class yaml_load():
    def __init__(self, cfg_path):
        import yaml
        with open(cfg_path, 'r') as f:
            kwargs = yaml.safe_load(f)
        for k, v in kwargs.items():
            setattr(self, k, v)
        if isinstance(self.lr, str):
            self.lr = eval(self.lr)

def train(args, model:torch.nn.Module, loss_fun, train_loader, val_loader):
    device = args.device
    de_parallel(model).train()
    # model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schduler = lr_scheduler.LinearLR(optimizer, 
                                     start_factor=1, 
                                     end_factor=args.lr_final, 
                                     total_iters=args.epochs)
    schduler_warmup = lr_scheduler.LinearLR(optimizer, 
                                            start_factor=0.001, 
                                            end_factor=1, 
                                            total_iters=args.warmup)
    scaler = GradScaler(device=device)
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for step, (images, texts) in enumerate(train_loader):
            step += epoch * len(train_loader)
            
            print('rank:%d ' % RANK, 'step:%d '%step, 'tttt: ', images.shape)
            #time.sleep(1)
            
            with autocast(device, enabled=args.amp, dtype=torch.bfloat16):
                output = model(images.to(device), texts.to(device))
                loss = loss_fun(**output)
                loss = loss * WORLD_SIZE
            
            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()            

            if step < args.warmup:
                schduler_warmup.step()
            
            if (step + 1) % args.accumulate == 0:
                if args.amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                optimizer.zero_grad()

                if is_master():
                    print_step(step, loss.data.item())
            
            # if (step + 1) % args.val_accumulate == 0:
            if step % args.val_accumulate == 0:
                with torch_distributed_zero_first(RANK): # barrier for no master
                    if is_master():
                        current_lr = optimizer.param_groups[0]['lr']
                        metric = {'step': step, 
                                'train_loss': loss.data.item(), 
                                'lr': current_lr}
                        metric.update(val(args, model, val_loader))
                        #save_checkpoint(model, step, args.save_dir)
                        save_metric(metric, args.metric_csv)
                
        if step >= args.warmup:
            schduler.step()

    # save_checkpoint(model, step, args.save_dir)

@torch.no_grad()
def val(args, model, val_loader):
    # metric: acc and avg smilarity 
    # val_loader for no shuffle
    device = args.device
    model.eval()
    metric = {'acc': 0, 'smilarity': 0}
    for i, (images, texts) in enumerate(val_loader):
        print('vvvvv: ', images.shape)
        # time.sleep(1)

        logits_per_image, _ = model(images.to(device), texts.to(device))
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        preds = logits_per_image.argmax(dim=-1).cpu().numpy()
        labels = np.arange(images.shape[0])
        acc = (preds == labels).sum() / len(labels)
        avg_smilarity = probs[labels, labels].mean()
        metric['acc'] = (metric['acc'] * i + acc) / (i + 1)
        metric['smilarity'] = (metric['smilarity'] * i + avg_smilarity) / (i + 1)
    model.train()
    return metric

if __name__ == '__main__':
    setup_ddp_envs()
    cfg = 'config.yaml'
    args = yaml_load(cfg)
    # update cfg
    args.val_accumulate = args.val_accumulate // WORLD_SIZE
    args.warmup = args.warmup // WORLD_SIZE
    args.num_workers = min(os.cpu_count(), args.num_workers * WORLD_SIZE)

    # build model
    model, transform = build_model_transform(args.model_hyp)
    if args.resum_path != '' and args.resum_path is not None:
        checkpoint = torch.jit.load(args.resum_path)
        model.load_state_dict(safe_state_dict(model, checkpoint.state_dict()))
        
    # build ddp
    if  WORLD_SIZE > 1:
        args.device = f'cuda:{RANK}'
        model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[RANK], 
                                                          output_device=RANK, 
                                                          find_unused_parameters=False)
    # build loss
    args.loss_hyp['rank'] = RANK
    args.loss_hyp['world_size'] = WORLD_SIZE 
    loss_fun = build_clip_loss(args.loss_hyp)

    # build data
    tokenizer = build_tokenizer()
    train_loader = bulid_dataloader_dist(args, args.train, transform, tokenizer, shuffle=True, rank=RANK)
    if is_master():
        val_loader = bulid_dataloader(args,  args.val, transform, tokenizer, shuffle=False)
    else:
        val_loader = None
    train(args, model, loss_fun, train_loader, val_loader)
    

    """ 
    python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 /home/chaofeng/clip_finetune/train_dist.py

    nohup /var/lib/anaconda3/envs/tkh/bin/python /home/chaofeng/clip_finetune/train.py > /home/chaofeng/clip_finetune/n_scratch_amp.log 2>&1 &
    """