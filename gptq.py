import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *

# import matplotlib.pyplot as plt
DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer,is_layered,sparsity,wbits):
        self.is_layered=is_layered
        self.wbits=wbits
        if self.is_layered:   
            self.layer = layer.Q_weight
            self.P_weight=layer.P_weight
        else:
            self.layer=layer
        self.hasq=False
        self.sparsity=sparsity
        self.currsample=0
        self.dev = self.layer.weight.device
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        del W
        torch.cuda.empty_cache()
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.mask1=torch.zeros(self.rows,self.columns).to("cpu")

    def add_batch(self, inp):
        
        self.currsample+=1
        if self.currsample<=128 :
            if DEBUG:
                self.inp1 = inp
                self.out1 = None
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()
            if isinstance(self.layer, nn.Conv2d):
                unfold = nn.Unfold(
                    self.layer.kernel_size,
                    dilation=self.layer.dilation,
                    padding=self.layer.padding,
                    stride=self.layer.stride
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            # inp = inp.float()
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            # self.H += 2 / self.nsamples * inp.matmul(inp.t())
            self.H += inp.matmul(inp.t())
            if self.currsample==128:
                #quantifying
                self.fasterquant()
                print("========================================")     
            raise ValueError
        if self.currsample>128:
            None
        
    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
    
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        #Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp#对角线上的元素都增加了 damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        del H
        torch.cuda.empty_cache()
        if self.is_layered:
            W_P = torch.zeros_like(W)


        
        #Calculate static weight importance
        temp_quantizer = Quantizer()
        temp_quantizer.configure(self.wbits, perchannel=True, sym=False, mse=True)
        temp_quantizer.find_params(W, weight=True)
        W_quant = temp_quantizer.quantize(W)
        
        losses0=torch.zeros_like(W)
        for ii in range(0,self.columns):
            d = Hinv[ii, ii]
            q0=W_quant[:,ii]
            w0=W[:,ii]
            losses0[:, ii] = (w0 - q0) ** 2 / d ** 2


        #Determine sparse mask
        percentile=self.sparsity
        flat_losses0 = losses0.view(-1)
        sorted_losses, _ = torch.sort(flat_losses0, descending=True)
        thresh = sorted_losses[int(len(sorted_losses) * (percentile / 100.0))]
        self.mask1.fill_(0)
        self.mask1.view(-1)[flat_losses0 >= thresh] = 1

        # tick1 = time.time()
        W_temp=W.to(device=self.dev)
        mask_temp=self.mask1.to(device=self.dev)
        # self.quantizer.find_params(W_temp*(1-mask_temp),weight=True)
        self.quantizer.find_params(W_temp,weight=True)
        # print('time 111%.2f' % (time.time() - tick1))
        del W_temp
        del mask_temp
        del sorted_losses
        del losses0
        del q0
        del w0
        
        torch.cuda.empty_cache()
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            # Q1 = torch.zeros_like(W1)
            # if self.is_layered:   
            #     W_P1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            #Losses1 = torch.zeros_like(W1).to('cpu')
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            mask1 = self.mask1[:, i1:i2].to(device=self.dev,dtype=torch.long)
            # print(sum(sum(mask1)))
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale.to(device=self.dev), self.quantizer.zero.to(device=self.dev), self.quantizer.maxq.to(device=self.dev)
                ).flatten()
                mask_index = torch.nonzero(mask1[:, i]).squeeze()
                q[mask_index] = w[mask_index]
                if self.is_layered:
                    Q[:,i1+i]=q*(1-mask1[:, i])
                    W_P[:,i1+i]=q*(mask1[:,i])
                else:
                    Q[:, i1+i] = q
                #Losses1[:, i] = (w - q) ** 2 / d** 2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            # if self.is_layered:
            #     W_P[:, i1:i2] = W_P1
            # Q[:, i1:i2] = Q1
            #Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        #print('error', torch.sum(Losses).item())

        
        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        if self.is_layered:
            self.P_weight.weight.data = W_P.reshape(self.P_weight.weight.shape).to(self.P_weight.weight.data.dtype)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))



    
    
    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
