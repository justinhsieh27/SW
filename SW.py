# --------------------------------------------------------
# Traditioanl Chinese word classification based on Swin-transformer
# Written by Justin Hsieh
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger

#20210602, Justin
#from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from utils import load_checkpoint, load_checkpoint2, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

#20210522, Justin
import heapq

import cv2

import PIL
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from IPython.display import display

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

from types import SimpleNamespace


def test():
    print("test")


class SWClassifier:
    def __init__(self, filename):
        
        args = SimpleNamespace()

        args.accumulation_steps=None
        args.amp_opt_level='O1'
        args.batch_size=None
        args.cache_mode='part'
        args.cfg='configs/swin_tiny_patch4_window7_224.yaml'
        args.data_path='zhTW_preprocess3'
        args.eval=True
        args.local_rank=0
        args.opts=['TRAIN.AUTO_RESUME', 'False']
        args.output='output'
        args.resume=None
        #args.resume2='ckpt_epoch_149.pth'
        args.resume2=filename
        args.tag=None
        args.throughput=False
        args.use_checkpoint=False
        args.zip=False


        config=get_config(args)
        #print("***config***")
        #print(config)
        #print("***config***")

        #environ({'CUDNN_VERSION': '8.0.4.30', '__EGL_VENDOR_LIBRARY_DIRS': '/usr/lib64-nvidia:/usr/share/glvnd/egl_vendor.d/', 'PYDEVD_USE_FRAME_EVAL': 'NO', 'LD_LIBRARY_PATH': '/usr/lib64-nvidia', 'CLOUDSDK_PYTHON': 'python3', 'LANG': 'en_US.UTF-8', 'HOSTNAME': '8ae9d729ac9b', 'OLDPWD': '/', 'CLOUDSDK_CONFIG': '/content/.config', 'NVIDIA_VISIBLE_DEVICES': 'all', 'DATALAB_SETTINGS_OVERRIDES': '{"kernelManagerProxyPort":6000,"kernelManagerProxyHost":"172.28.0.3","jupyterArgs":["--ip=\\"172.28.0.2\\""],"debugAdapterMultiplexerPath":"/usr/local/bin/dap_multiplexer"}', 'ENV': '/root/.bashrc', 'PAGER': 'cat', 'NCCL_VERSION': '2.7.8', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'JPY_PARENT_PID': '61', 'NO_GCE_CHECK': 'True', 'PWD': '/content/drive/My Drive/Swin-Transformer', 'HOME': '/root', 'LAST_FORCED_REBUILD': '20210528', 'CLICOLOR': '1', 'DEBIAN_FRONTEND': 'noninteractive', 'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs', 'GCE_METADATA_TIMEOUT': '0', 'GLIBCPP_FORCE_NEW': '1', 'TBE_CREDS_ADDR': '172.28.0.1:8008', 'SHELL': '/bin/bash', 'TERM': 'xterm-color', 'GCS_READ_CACHE_BLOCK_SIZE_MB': '16', 'PYTHONWARNINGS': 'ignore:::pip._internal.cli.base_command', 'MPLBACKEND': 'module://ipykernel.pylab.backend_inline', 'CUDA_VERSION': '11.0.3', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'SHLVL': '1', 'PYTHONPATH': '/env/python', 'NVIDIA_REQUIRE_CUDA': 'cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451', 'COLAB_GPU': '1', 'GLIBCXX_FORCE_NEW': '1', 'PATH': '/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin', 'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4', 'GIT_PAGER': 'cat', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '12345', 'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'})
        os.environ= {'CUDNN_VERSION': '8.0.4.30', '__EGL_VENDOR_LIBRARY_DIRS': '/usr/lib64-nvidia:/usr/share/glvnd/egl_vendor.d/', 'PYDEVD_USE_FRAME_EVAL': 'NO', 'LD_LIBRARY_PATH': '/usr/lib64-nvidia', 'CLOUDSDK_PYTHON': 'python3', 'LANG': 'en_US.UTF-8', 'HOSTNAME': '8ae9d729ac9b', 'OLDPWD': '/', 'CLOUDSDK_CONFIG': '/content/.config', 'NVIDIA_VISIBLE_DEVICES': 'all', 'DATALAB_SETTINGS_OVERRIDES': '{"kernelManagerProxyPort":6000,"kernelManagerProxyHost":"172.28.0.3","jupyterArgs":["--ip=\\"172.28.0.2\\""],"debugAdapterMultiplexerPath":"/usr/local/bin/dap_multiplexer"}', 'ENV': '/root/.bashrc', 'PAGER': 'cat', 'NCCL_VERSION': '2.7.8', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true', 'JPY_PARENT_PID': '61', 'NO_GCE_CHECK': 'True', 'PWD': '/content/drive/My Drive/Swin-Transformer', 'HOME': '/root', 'LAST_FORCED_REBUILD': '20210528', 'CLICOLOR': '1', 'DEBIAN_FRONTEND': 'noninteractive', 'LIBRARY_PATH': '/usr/local/cuda/lib64/stubs', 'GCE_METADATA_TIMEOUT': '0', 'GLIBCPP_FORCE_NEW': '1', 'TBE_CREDS_ADDR': '172.28.0.1:8008', 'SHELL': '/bin/bash', 'TERM': 'xterm-color', 'GCS_READ_CACHE_BLOCK_SIZE_MB': '16', 'PYTHONWARNINGS': 'ignore:::pip._internal.cli.base_command', 'MPLBACKEND': 'module://ipykernel.pylab.backend_inline', 'CUDA_VERSION': '11.0.3', 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility', 'SHLVL': '1', 'PYTHONPATH': '/env/python', 'NVIDIA_REQUIRE_CUDA': 'cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451', 'COLAB_GPU': '1', 'GLIBCXX_FORCE_NEW': '1', 'PATH': '/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin:/opt/bin', 'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4', 'GIT_PAGER': 'cat', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': '12345', 'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}
        print(os.environ)


        if config.AMP_OPT_LEVEL != "O0":
            assert amp is not None, "amp not installed!"

        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        else:
            rank = -1
            world_size = -1


        torch.cuda.set_device(config.LOCAL_RANK)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.distributed.barrier()

        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

        # linear scale the learning rate according to total batch size, may not be optimal
        linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
        # gradient accumulation also need to scale the learning rate
    
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
            linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

        os.makedirs(config.OUTPUT, exist_ok=True)
        logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())

        print("***config***")
        print(config)
        print("***config***")

        dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
        model = build_model(config)
        model.cuda()
        logger.info(str(model))

        optimizer = build_optimizer(config, model)
        if config.AMP_OPT_LEVEL != "O0":
            model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

        if config.AUG.MIXUP > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif config.MODEL.LABEL_SMOOTHING > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        max_accuracy = 0.0

        # 20210601, Justin    
        max_accuracy = load_checkpoint2(config, model_without_ddp, logger)
        self.model, self.dicClass, self.loader = self.validate2(config, data_loader_val, model, dataset_val, dataset_train)
    
        return



    #20210602, Justin
    @torch.no_grad()
    def validate2(self, config, data_loader, model, dataset_val, dataset_train):
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()

        # 20210502, Justin
        dicClass=dataset_train.class_to_idx
        dicClass = {v: k for k, v in dicClass.items()} # reverse dic
        print(dicClass)



        #20210603, Justin
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        # loader使用torchvision中自带的transforms函数
        loader = transforms.Compose([transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])  
        unloader = transforms.ToPILImage()
    
        return model, dicClass, loader




    # Remove background
    def RemBack(self, img):

        maxPixel = 0
        minPixel = 255
    
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                mean = sum(img[i][j])/3.0

                if mean > maxPixel:
                    maxPixel = mean
                if mean < minPixel:
                    minPixel = mean
                
        backThreshold = int(minPixel + (maxPixel - minPixel)*0.65)

        #backThreshold = 160
        #backThreshold = 200

        #print(img.shape)

        img_Remback = img.copy()

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                mean = sum(img[i][j])/3.0
                if mean > backThreshold:
                    img_Remback[i][j]=[255, 255, 255]
    
        return img_Remback


    # Remove red line
    def RemRedline(self, img):
        redRatioTh = 0.38  # red ratio threshold
        redValueTh = 32   # red absolute value threshold

        img_Remred = img.copy()

        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                redRatio = img[i][j][0] / (sum(img[i][j]) + 0.1)
                if redRatio > redRatioTh and img[i][j][0] > redValueTh :
                    img_Remred[i][j]=[255, 255, 255]

        return img_Remred






    #def Inference(self, filename):
    def Inference(self, img):
        #20210603, Justin
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        #loader使用torchvision中自带的transforms函数
        #loader = transforms.Compose([transforms.Resize((224, 224), interpolation=PIL.Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])  
        #unloader = transforms.ToPILImage()

        #print(filename)

        # opencv open file, and then convert to PIL
        #img = cv2.imread(filename)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        #preprocessing
        img = self.RemBack(img)
        img = self.RemRedline(img)
        
        
        tensorImage =Image.fromarray(img,mode="RGB")

        # PIL open image file
        #tensorImage = Image.open(filename).convert('RGB')
    
        tensorImage = self.loader(tensorImage).unsqueeze(0)
        tensorImage = tensorImage.to('cpu', torch.float)

        #print(tensorImage.shape)
        #print(tensorImage)

        tensorImage = tensorImage.cuda(non_blocking=True)
        #print(tensorImage)

        # compute output
        output = self.model(tensorImage)
        
        
        #print(output.size())
        lstResulByCls = []
        for i in range(output.size()[1]):
            lstResulByCls.append(output[0][i].data.tolist())
        #print(lstResulByCls)
        
        lstTop5Index = list(map(lstResulByCls.index, heapq.nlargest(5, lstResulByCls)))
        print(lstTop5Index)
        
        lstResult = []
        # list the top5 probability
        for i in lstTop5Index:
            print(self.dicClass[i] + ": " + str(lstResulByCls[i]))
            lstResult.append([self.dicClass[i], lstResulByCls[i]])
    
        #return lstResult

        if lstResult[0][1] < 4.5:
            return "isnull"
        else:
            return lstResult[0][0]


