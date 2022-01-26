import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
'''more datasets'''
from ltr.dataset import Got10k, Lasot, MSCOCOSeq, ImagenetVID, ImagenetDET, Youtube_VOS, Saliency
from ltr.data import processing, LTRLoader
import ltr.models.bbreg.atom as atom_models
from ltr import actors
from ltr.trainers import LTRTrainer
import ltr.data.transforms as dltransforms
'''newly added'''
from ltr.data import sampler_mask # 能够与mask数据兼容的格式
import torch
from torch.utils.data.distributed import DistributedSampler

def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'ATOM IoUNet with larger datasets.'
    settings.print_interval = 100                                 # How often to print loss and other info
    settings.batch_size = 64                                    # Batch size
    settings.num_workers = 4                                    # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 5.0                           # Image patch size relative to target size
    settings.feature_sz = 18                                    # Size of feature map
    settings.output_sz = settings.feature_sz * 16               # Size of input image patches

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 4.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    settings.proposal_params = {'min_iou': 0.1, 'boxes_per_frame': 16, 'sigma_factor': [0.01, 0.05, 0.1, 0.2, 0.3]}

    # Train datasets
    '''Use more datasets'''
    got_10k_train = Got10k(settings.env.got10k_dir, split='train')
    lasot_train = Lasot(split='train')
    coco_train = MSCOCOSeq()
    imagenet_vid = ImagenetVID()
    imagenet_det = ImagenetDET()
    # mask datasets
    youtube_vos = Youtube_VOS()
    saliency = Saliency()
    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)
    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean,
                                                                                       std=settings.normalize_std)])
    # Data processing to do on the training pairs
    data_processing_train = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      proposal_params=settings.proposal_params,
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)
    # The sampler for training
    '''use SEMasksampler due to mask datasets'''
    dataset_train = sampler_mask.ATOMSampler([lasot_train, got_10k_train, coco_train, imagenet_vid, imagenet_det,
                                         youtube_vos, saliency], [1, 1, 1, 1, 1, 2, 3],
                                        samples_per_epoch=1000 * settings.batch_size, max_gap=50,
                                        processing=data_processing_train)
    '''multiple gpu training'''
    train_sampler = DistributedSampler(dataset_train)
    '''"sampler" is exclusive with "shuffle"'''
    # The loader for training
    # loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
    #                          num_workers=settings.num_workers,
    #                          shuffle=True, drop_last=True, stack_dim=1)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size,
                             num_workers=settings.num_workers,
                             drop_last=True, stack_dim=1, sampler=train_sampler, pin_memory=False)

    # Validation datasets
    lasot_test = Lasot(split='test')
    # The augmentation transform applied to the validation set (individually to each image in the pair)
    transform_val = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])
    # Data processing to do on the validation pairs
    data_processing_val = processing.ATOMProcessing(search_area_factor=settings.search_area_factor,
                                                    output_sz=settings.output_sz,
                                                    center_jitter_factor=settings.center_jitter_factor,
                                                    scale_jitter_factor=settings.scale_jitter_factor,
                                                    mode='sequence',
                                                    proposal_params=settings.proposal_params,
                                                    transform=transform_val,
                                                    joint_transform=transform_joint)
    # The sampler for validation
    dataset_val = sampler_mask.ATOMSampler([lasot_test], [1], samples_per_epoch=500*settings.batch_size, max_gap=50,
                                      processing=data_processing_val)
    # The loader for validation
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=settings.batch_size, num_workers=settings.num_workers,
                           shuffle=False, drop_last=True, epoch_interval=5, stack_dim=1)

    # Create network
    '''Using ResNet-50'''
    net = atom_models.atom_resnet50(backbone_pretrained=True)
    '''multiple gpu training'''
    net = net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[settings.local_rank])
    # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[settings.local_rank],find_unused_parameters=True)

    # Set objective
    objective = nn.MSELoss()

    # Create actor, which wraps network and objective
    actor = actors.AtomActor(net=net, objective=objective)

    # Optimizer
    # optimizer = optim.Adam(actor.net.bb_regressor.parameters(), lr=1e-3)
    optimizer = optim.Adam(actor.net.parameters(), lr=1e-3)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    # Create trainer
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(40, load_latest=True, fail_safe=False)
