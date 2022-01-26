import torch.nn as nn
import ltr.models.backbone as backbones
from ltr.models.neck import CorrNL,Depth_Corr,Naive_Corr
from ltr.models.head import bbox,corner,mask
from ltr import model_constructor

class SEbcmnet(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """
    def __init__(self, feature_extractor, neck_module, head_module, used_layers,
                 extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEbcmnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.bbox_head, self.corner_head, self.mask_head = head_module
        self.used_layers = used_layers

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)
        if unfreeze_layer3:
            for p in self.feature_extractor.layer3.parameters():
                p.requires_grad_(True)
    def forward(self, train_imgs, test_imgs, train_bb, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        self.forward_ref(train_imgs, train_bb)
        pred_dict = self.forward_test(test_imgs, mode)
        return pred_dict

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass of reference branch.
        size of train_imgs is (1,batch,3,H,W), train_bb is (1,batch,4)"""
        '''train_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        num_sequences = train_imgs.shape[-4] # batch
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat的数据类型都是OrderedDict,字典的键为'layer3' '''
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:])) # 输入size是(batch,3,256,256)

        train_feat_list = [feat for feat in train_feat_dict.values()] #list,其中每个元素对应一层输出的特征(tensor)

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))


    def forward_test(self, test_imgs, mode='train'):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        output = {}
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])# 输入size是(batch,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        # Save low-level feature list
        Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat([test_feat_dict['layer3']])
        # Obtain bbox prediction
        if mode=='train':
            output['bbox'] = self.bbox_head(fusion_feat)
            output['corner'] = self.corner_head(fusion_feat)
            output['mask'] = self.mask_head(fusion_feat,Lfeat_list)
        elif mode=='test':
            output['feat'] = fusion_feat
            output['bbox'] = self.bbox_head(fusion_feat)
            output['corner'] = self.corner_head(fusion_feat)
            output['mask'] = self.mask_head(fusion_feat, Lfeat_list)
        else:
            raise ValueError("mode should be train or test")
        return output

    def get_test_feat(self, test_imgs):
        """ Forward pass of test branch. size of test_imgs is (1,batch,3,256,256)"""
        # Extract backbone features
        '''test_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                        layers=['conv1','layer1','layer2','layer3'])# 输入size是(batch,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        # Save low-level feature list
        self.Lfeat_list = [feat.contiguous() for name, feat in test_feat_dict.items() if name != 'layer3']

        # fuse feature from two branches
        self.fusion_feat = self.neck.fuse_feat([test_feat_dict['layer3']]).contiguous()
        return self.fusion_feat
    def get_output(self,mode):
        if mode == 'bbox':
            return self.bbox_head(self.fusion_feat)
        elif mode == 'corner':
            return self.corner_head(self.fusion_feat)
        elif mode == 'mask':
            return self.mask_head(self.fusion_feat, self.Lfeat_list)
        else:
            raise ValueError ('mode should be bbox or corner or mask')
    def get_corner_heatmap(self):
        return self.corner_head.get_heatmap(self.fusion_feat)
    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def SEbcm_resnet50(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # multiple heads
    bbox_head = bbox.BBox_Predictor(inplanes=pool_size*pool_size) # 64
    corner_head = corner.Corner_Predictor(inplanes=pool_size*pool_size) # 64
    # mask_head = mask.Mask_Predictor(inplanes=pool_size * pool_size)
    mask_head = mask.Mask_Predictor_fine()
    '''layer3 feature size=(64,1024,16,16), stride=16
    we temporally only use feature from layer3(3rd residual block) '''
    # net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net, head_module=(bbox_head,corner_head,mask_head),
    #               used_layers=used_layers, extractor_grad=False,unfreeze_layer3=unfreeze_layer3)
    '''尝试放开所有参数'''
    net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net,
                   head_module=(bbox_head, corner_head, mask_head),
                   used_layers=used_layers, extractor_grad=True, unfreeze_layer3=unfreeze_layer3)
    return net
'''ablation study (without Non-local)'''
@model_constructor
def SEbcm_resnet50_woNL(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    '''2020.2.23 not use non-local'''
    neck_net = CorrNL.CorrNL(pool_size=pool_size,use_NL=False)
    # multiple heads
    bbox_head = bbox.BBox_Predictor(inplanes=pool_size*pool_size)
    corner_head = corner.Corner_Predictor(inplanes=pool_size*pool_size)
    # mask_head = mask.Mask_Predictor(inplanes=pool_size * pool_size)
    mask_head = mask.Mask_Predictor_fine()
    '''layer3 feature size=(64,1024,16,16), stride=16
    we temporally only use feature from layer3(3rd residual block) '''
    # net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net, head_module=(bbox_head,corner_head,mask_head),
    #               used_layers=used_layers, extractor_grad=False,unfreeze_layer3=unfreeze_layer3)
    '''尝试放开所有参数'''
    net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net,
                   head_module=(bbox_head, corner_head, mask_head),
                   used_layers=used_layers, extractor_grad=True, unfreeze_layer3=unfreeze_layer3)
    return net
'''ablation study (Use depth-wise correlation)'''
@model_constructor
def SEbcm_resnet50_Depth(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    '''use Depth-wise correlation'''
    neck_net = Depth_Corr.Depth_Corr(pool_size=pool_size)
    # multiple heads
    bbox_head = bbox.BBox_Predictor(inplanes=pool_size*pool_size) # 64
    corner_head = corner.Corner_Predictor(inplanes=pool_size*pool_size) # 64
    # mask_head = mask.Mask_Predictor(inplanes=pool_size * pool_size)
    mask_head = mask.Mask_Predictor_fine()
    '''layer3 feature size=(64,1024,16,16), stride=16
    we temporally only use feature from layer3(3rd residual block) '''
    # net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net, head_module=(bbox_head,corner_head,mask_head),
    #               used_layers=used_layers, extractor_grad=False,unfreeze_layer3=unfreeze_layer3)
    '''尝试放开所有参数'''
    net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net,
                   head_module=(bbox_head, corner_head, mask_head),
                   used_layers=used_layers, extractor_grad=True, unfreeze_layer3=unfreeze_layer3)
    return net
'''ablation study (Use naive correlation)'''
@model_constructor
def SEbcm_resnet50_Naive(backbone_pretrained=True,used_layers=('layer3',),pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    '''use Depth-wise correlation'''
    neck_net = Naive_Corr.Naive_Corr(pool_size=pool_size)
    # multiple heads
    bbox_head = bbox.BBox_Predictor(inplanes=pool_size*pool_size) # 64
    corner_head = corner.Corner_Predictor(inplanes=pool_size*pool_size) # 64
    # mask_head = mask.Mask_Predictor(inplanes=pool_size * pool_size)
    mask_head = mask.Mask_Predictor_fine()
    '''尝试放开所有参数'''
    net = SEbcmnet(feature_extractor=backbone_net, neck_module=neck_net,
                   head_module=(bbox_head, corner_head, mask_head),
                   used_layers=used_layers, extractor_grad=True, unfreeze_layer3=unfreeze_layer3)
    return net





'''maybe  we can try ResNext, SENet... in the future'''