import torch.nn as nn
import torch.optim as optim
import torchvision.transforms

from ltr.dataset import Got10k, Lasot, MSCOCOSeq
from ltr.data import SEprocessing, sampler, LTRLoader
import ltr.data.transforms as dltransforms
import ltr.models.SEbb.SEbb as SEbb
from ltr import actors
from ltr.trainers import LTRTrainer
from ltr.models.loss.iou_loss import IOULoss

'''和SEbb_anchor相比,这个脚本把setting的center_jitter_factor跟scale_jitter_factor都缩小到了0.25
因为我把训练样本可视化出来,发现好多框都已经超出图像范围了(导致出现小于0或者大于256的坐标值),但是这些坐标值并没有被妥善处理,这次我要改过来
并且考虑到实际跟踪时不一定会出现这么极端的尺度变化或者中心偏移,所以我把这两项也调小了'''
def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'SEbb with default settings.'
    settings.print_interval = 1                                 # How often to print loss and other info
    settings.batch_size = 64                                    # Batch size
    settings.num_workers = 4                                    # Number of workers for image loading
    '''RGB order'''
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    '''In my design, scale factor is 2.0 rather than 5.0'''
    settings.search_area_factor = 2.0                           # Image patch size relative to target size
    settings.feature_sz = 16                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches
    settings.used_layers = ['layer3']

    # Settings for the image sample and proposal generation
    '''由于ATOM的搜索区域很大(5x),所以它的center_jitter_factor也很大(4.5)；
    但是我们的搜索区域是很小的(2x),所以我们的center_jitter_factor不能设那么大,不然的话
    “真值框可能并不落在搜索区域里”,所以我暂时改成0.5了,经过测试,现在不会再出现segmentation fault了
    这个值未来可以再调一调'''
    settings.center_jitter_factor = {'train': 0, 'test': 0.25}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}

    '''##### Prepare data for training and validation #####'''
    # Train datasets
    got_10k_train = Got10k(settings.env.got10k_dir, split='train')
    # lasot_train = Lasot(split='train')
    coco_train = MSCOCOSeq()

    # Validation datasets
    got_10k_val = Got10k(split='val')
    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = SEprocessing.SEProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # # Data processing to do on the validation pairs
    data_processing_val = SEprocessing.SEProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)

    # The sampler for training
    '''Build training dataset!!! focus "__getitem__" and "__len__"'''
    dataset_train = sampler.ATOMSampler([got_10k_train,coco_train], [1,1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)

    # The sampler for validation
    dataset_val = sampler.ATOMSampler([got_10k_val], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                      processing=data_processing_val)

    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)
    '''##### prepare network and other stuff for optimization #####'''
    # Create network
    '''特别注意: 这块要换成我们自己的'''
    net = SEbb.SEbb_resnet50_anchor(backbone_pretrained=True,used_layers=settings.used_layers,
                             pool_size = int(settings.feature_sz/2)) # 目标尺寸是输出特征尺寸的一半

    # Set objective
    '''GioU Loss'''
    objective = IOULoss(loss_type='giou')
    # objective = nn.MSELoss()
    # Create actor, which wraps network and objective
    actor = actors.SEbb_anchor_Actor(net=net, objective=objective)

    # Optimizer
    optimized_module = nn.Sequential(actor.net.neck, actor.net.head)
    optimizer = optim.Adam(optimized_module.parameters(), lr=1e-3)
    # optimizer = optim.Adam(optimized_module.parameters(), lr=1e-2)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)
    '''##### training #####'''
    # Run training (set fail_safe=False if you are debugging)
    trainer.train(40, load_latest=True, fail_safe=False)

