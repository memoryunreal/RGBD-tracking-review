import torch.nn as nn
import ltr.models.backbone as backbones
'''in this model, we only use mask branch'''
from ltr.models.neck import CorrNL
from ltr.models.head import mask
from ltr import model_constructor

class SEmask_fine(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """
    def __init__(self, feature_extractor, neck_module, head_module, used_layers=None, extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEmask_fine, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.mask_head = head_module
        self.used_layers = used_layers

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

        '''2019.12.30 新加的功能: 支持unfreeze一些层'''
        if unfreeze_layer3:
            for p in self.feature_extractor.layer3.parameters():
                p.requires_grad_(True)
    def forward(self, train_imgs, test_imgs, train_bb):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch"""
        '''train_imgs: (1,64,3,256,256), train_bb: (1,64,4)
        train_feat的数据类型是OrderedDict,字典的键为'layer3' '''
        num_sequences = train_imgs.shape[-4] # 64
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]),layers=['layer3'])# 输入size是(64,3,256,256)
        train_feat_list = [feat for feat in train_feat_dict.values()]

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))


    def forward_test(self, test_imgs):
        """ Forward pass of test branch"""
        '''test_imgs: (1,64,3,256,256)
        test_feat的数据类型都是OrderedDict,字典的键为'conv1','layer1','layer2' '''
        # Extract backbone features
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])# 输入size是(64,3,256,256)
        # Save low-level feature list
        Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']
        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([test_feat_dict['layer3']]) # [(1,3,16,16)] or [(64,3,16,16)]
        # Obtain mask prediction
        output = self.mask_head(fusion_feat, Lfeat_list)
        return output

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def SEmask_fine_resnet50(backbone_pretrained=True,pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # multiple heads
    mask_head = mask.Mask_Predictor_fine()

    net = SEmask_fine(feature_extractor=backbone_net, neck_module=neck_net, head_module=mask_head,
                  extractor_grad=False,unfreeze_layer3=unfreeze_layer3)

    return net

'''maybe  we can try ResNext, SENet... in the future'''