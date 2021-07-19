# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer,AutoModel,BertForMaskedLM,AutoModelForMaskedLM,BertModel,BertTokenizer
from transformers.utils.dummy_pt_objects import PretrainedBartModel
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
import  torch

class ClassifierLit(pl.LightningModule):
    """构建分类模型，借助预训练的模型　
    　
    """

    def __init__(self,learning_rate=3e-4,warmup=1,num_labels=2,frequency=1,patience=50,T_max=10,verbose=True,Pretrained="hfl/rbtl3",forceBert=False):
        """[summary]
        初始化模型，主要用来

        Args:
            learning_rate ([type], optional): [description]. Defaults to 3e-4.
            warmup (int, optional): [description]. Defaults to 1.
            num_labels (int, optional): [description]. Defaults to 2.
            frequency (int, optional): [description]. Defaults to 1.
            patience (int, optional): [description]. Defaults to 50.
            T_max (int, optional): [description]. Defaults to 10.
            verbose (bool, optional): [description]. Defaults to True.
            Pretrained : 预训练模型
            forceBert : 强制使用ｂｅｒｔ加载　默认为False
        """
        
        super().__init__()
        self.num_labels=num_labels

#         self.model = AutoModel.from_pretrained("hfl/chinese-electra-180g-small-generator") 
#         self.classifier = torch.nn.Linear(in_features=self.model.embeddings_project.out_features, out_features=self.num_labels)
        if forceBert:
            self.model = BertModel.from_pretrained(Pretrained) 
        else:
            self.model = AutoModel.from_pretrained(Pretrained) 
         
        self.classifier = torch.nn.Linear(in_features=self.model.pooler.dense.out_features, out_features=self.num_labels)
#         self.model.pooler.dense=torch.nn.Linear(in_features=768, out_features=self.num_labels, bias=True)
    
    
        self.dropout = nn.Dropout(0.1)
        self.learning_rate=learning_rate
        self.warmup=warmup
        self.frequency=frequency # 每多少次检查变化
        self.patience=patience #多少次不变化后更新
        self.T_max=T_max # 默认设为step重置步数
        self.verbose=verbose
        # del self.bert
    def forward(self, x,attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        y=None):
        """[summary]
        训练
        
        Args:
            x ([type]): [description]
            y ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        # in lightning, forward defines the prediction/inference actions
        pooler_output = self.model(input_ids=x,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   token_type_ids=token_type_ids).pooler_output
        pooler_output = self.dropout(pooler_output) # 获取第一个输出
        logits=self.classifier(pooler_output)
        if y==None:
            return logits
        loss = None
        if self.num_labels == 1:
            #  We are doing regression
            loss_fct = torch.nn.MSELoss()
            loss = loss_fct(logits.view(-1), y.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), y.view(-1))  
        return logits,loss

    def training_step(self, batch, batch_idx):
        """[summary]
        
        处理单个批次的数据

        Args:
            batch ([type]): [description]
            batch_idx ([type]): [description]
        """

        # training_step defined the train loop.
        # It is independent of forward
#         if self.global_step in [101,1002,10000]:
#             self.configure_optimizers()
            
        pred,loss=self(batch[0],token_type_ids=batch[1],attention_mask=batch[2],y=batch[-1])
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        """一次batch训练"""
        # training_step defined the train loop.
        # It is independent of forward
        pred,loss=self(batch[0],token_type_ids=batch[1],attention_mask=batch[2],y=batch[-1])
        
        acc = FM.accuracy(torch.softmax(pred,dim=1), batch[1])
        # Logging to TensorBoard by default
#         self.log('test_loss', loss)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics
    def test_step(self, batch, batch_idx):
        """一次batch训练"""
        # training_step defined the train loop.
        # It is independent of forward
        
        pred,loss=self(batch[0],token_type_ids=batch[1],attention_mask=batch[2],y=batch[-1])
        
        acc = FM.accuracy(torch.softmax(pred,dim=1), batch[1])
        # Logging to TensorBoard by default
#         self.log('test_loss', loss)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics
    
    def on_train_epoch_start(self):
        """开始训练执行"""
#         if self.current_epoch in [5,50,100]:
            
#             self.configure_optimizers()
        pass
    def configure_optimizers(self):
        """从下面几个优化方案里选择"""
        
#         op1=self.configure_optimizers_v2()
#         op2=self.configure_optimizers_v4()
        print("global_step",self.global_step)
#         if self.current_epoch >=0:
#             return self.configure_optimizers_v2()
#         elif 51> self.current_epoch >=6:
#             return self.configure_optimizers_v3()
#         else:
#             return self.configure_optimizers_v4()
#         return [op1[0],op2[0]],[op1[1],op2[1]]
        return self.configure_optimizers_v4()
    
    def get_optimizer(self):
        # print("global_step",self.global_step)
        if self.global_step >=1001:
            print("SGD")
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            print("AdamW")
            optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))
#         optimizer = torch.optim.Adadelta(self.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
        return optimizer
    def configure_optimizers_v1(self):
        """优化器"""
        optimizer = self.get_optimizer()
        
        #https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html#learning-rate-scheduling
        def warm_decay(step):
            if step < self.warmup:
                return  step / self.warmup
            if self.verbose:
                print("step",step,self.warmup ** 0.5 * step ** -0.5)
            return self.warmup ** 0.5 * step ** -0.5

        lr_scheduler={
        'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer,warm_decay),
        "interval": "step", #runs per batch rather than per epoch
        "frequency": self.frequency,
        'name': 'lr_scheduler'
        }
#         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_optimizers_v2(self):
        """优化器 自动优化器"""
        optimizer = self.get_optimizer()
        #         使用自适应调整模型
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=self.patience,factor=0.1,verbose=self.verbose)

#         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119
        
        lr_scheduler={
#            'optimizer': optimizer,
           'scheduler': scheduler,
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'interval': 'step',
            'frequency': self.frequency,
            'name':"lr_scheduler",
            'monitor': 'train_loss', #监听数据变化
            'strict': True,
        }
#         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    def configure_optimizers_v3(self):
        """优化器 余玄退火方案 https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR"""
        optimizer = self.get_optimizer()
        #         使用自适应调整模型
        
        #  , verbose=True
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.T_max, eta_min=0, last_epoch=-1,verbose=self.verbose)

#         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119
        
        lr_scheduler={
#            'optimizer': optimizer,
           'scheduler': scheduler,
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'interval': 'step',
            'frequency': self.frequency,
            'name':"lr_scheduler",
            'monitor': 'train_loss', #监听数据变化
            'strict': True,
        }
#         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_optimizers_v4(self):
        """优化器 # 类似于余弦，但其周期是变化的，初始周期为T_0,而后周期会✖️T_mult。每个周期学习率由大变小； https://www.notion.so/62e72678923f4e8aa04b73dc3eefaf71"""
#         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))

        #只优化部分
#         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate))
# https://pytorch.org/docs/stable/optim.html#torch.optim.Adadelta
        optimizer = self.get_optimizer()
        #         使用自适应调整模型
        T_mult=2
        scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=self.T_max,T_mult=T_mult,eta_min=0 ,verbose=self.verbose)
#         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119
        
        lr_scheduler={
#            'optimizer': optimizer,
           'scheduler': scheduler,
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'interval': 'epoch', #epoch/step
            'frequency': self.frequency,
            'name':"lr_scheduler",
            'monitor': 'train_loss', #监听数据变化
            'strict': True,
        }
#         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
#     def configure_optimizers_v5(self):
#         """失败 AdamW 问题 优化器 # # 类似于余弦，但其周期是变化的，初始周期为T_0,而后周期会✖️T_mult。每个周期学习率由大变小； https://www.notion.so/62e72678923f4e8aa04b73dc3eefaf71"""
#         optimizer = torch.optim.AdamW(self.parameters(), lr=(self.learning_rate),momentum=0.9)
#         #         使用自适应调整模型

# #         T_mult=int(self.T_max/5)
#         T_mult=2
# #         scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=self.T_max,T_mult=T_mult, verbose=True)
#         scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-8,max_lr=self.learning_rate,step_size_up=self.T_max,mode="triangular2")  
# #         https://github.com/PyTorchLightning/pytorch-lightning/blob/6dc1078822c33fa4710618dc2f03945123edecec/pytorch_lightning/core/lightning.py#L1119
        
#         lr_scheduler={
# #            'optimizer': optimizer,
#            'scheduler': scheduler,
#             'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
#             'interval': 'step',
#             'frequency': self.frequency,
#             'name':"lr_scheduler",
#             'monitor': 'train_loss', #监听数据变化
#             'strict': True,
#         }
#         return [optimizer], [lr_scheduler]
