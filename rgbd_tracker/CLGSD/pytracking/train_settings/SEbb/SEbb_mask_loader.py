import torchvision.transforms
from ltr.dataset import Youtube_VOS,Saliency

from ltr.data import SEprocessing, SEsampler, LTRLoader
import ltr.data.transforms as dltransforms


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.description = 'test dataloader for mask dataset.'
    settings.batch_size = 128
    settings.num_workers = 4                                    # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]              # Normalize std (default pytorch ImageNet values)
    settings.search_area_factor = 2.0                           # Image patch size relative to target size
    settings.output_sz = 16*16
    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 0.25}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}
    '''##### Prepare data for training and validation #####'''
    # Train datasets
    youtube_vos = Youtube_VOS()
    # saliency = Saliency()
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
    dataset_train = SEsampler.SEMaskSampler([youtube_vos], [1],
                                        samples_per_epoch=1000*settings.batch_size, max_gap=50,
                                        processing=data_processing_train)

    # The loader for training
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=settings.batch_size, num_workers=settings.num_workers,
                             shuffle=True, drop_last=True, stack_dim=1)
    return loader_train


