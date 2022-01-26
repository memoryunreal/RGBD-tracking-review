
import torchvision.transforms

from ltr.dataset import Got10k
# from ltr.dataset import Got10k, Lasot, MSCOCOSeq
# from ltr.dataset import ImagenetDET
# from ltr.dataset import ImagenetVID

from ltr.data import SEprocessing, SEsampler, LTRLoader
import ltr.data.transforms as dltransforms

'''使用更大的搜索区域(4倍大),更大的偏移量(0.5),更大的尺寸变化(0.5)'''
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
    settings.center_jitter_factor = {'train': 0, 'test': 0.5}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.5}
    '''##### Prepare data for training and validation #####'''
    # Train datasets
    # coco = MSCOCOSeq()
    # lasot = Lasot(split='test')
    got_10k_train = Got10k(settings.env.got10k_dir, split='train')
    # The joint augmentation transform, that is applied to the pairs jointly
    transform_joint = dltransforms.ToGrayscale(probability=0.05)

    # The augmentation transform applied to the training set (individually to each image in the pair)
    transform_train = torchvision.transforms.Compose([dltransforms.ToTensorAndJitter(0.2),
                                                      torchvision.transforms.Normalize(mean=settings.normalize_mean, std=settings.normalize_std)])

    # Data processing to do on the training pairs
    data_processing_train = SEprocessing.SEMaskProcessing(search_area_factor=settings.search_area_factor,
                                                      output_sz=settings.output_sz,
                                                      center_jitter_factor=settings.center_jitter_factor,
                                                      scale_jitter_factor=settings.scale_jitter_factor,
                                                      mode='sequence',
                                                      transform=transform_train,
                                                      joint_transform=transform_joint)

    # The sampler for training
    '''Build training dataset!!! focus "__getitem__" and "__len__"'''
    dataset_train = SEsampler.SEMaskSampler([got_10k_train], [1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)
    return loader_train


