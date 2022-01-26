import torch.nn as nn
import ltr.models.backbone as backbones
'''new design: neck and head'''
from ltr.models.neck import CorrNL
from ltr.models.head import bbox
from ltr import model_constructor

class SEbbnet(nn.Module):
    """ Scale Estimation network module"""
    def __init__(self, feature_extractor, neck_module, head_module, used_layers, extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEbbnet, self).__init__()

        self.feature_extractor = feature_extractor
        self.neck = neck_module
        self.head = head_module
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
        '''train_imgs: (1,64,3,256,256), test_imgs: (1,64,3,256,256), train_bb: (1,64,4)'''
        num_sequences = train_imgs.shape[-4] # 64
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat与test_feat的数据类型都是OrderedDict,字典的键为'layer2','layer3'等等'''
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))   # 输入size是(64,3,256,256)
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))      # 输入size是(64,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        ''' layer2的特征size为(64,512,32,32),  stride=8
            layer3的特征size为(64,1024,16,16), stride=16(我想先只用layer3)'''
        train_feat_list = [feat for feat in train_feat_dict.values()]
        test_feat_list = [feat for feat in test_feat_dict.values()]

        # fuse feature from two branches
        fusion_feat = self.neck(train_feat_list, test_feat_list,
                                train_bb.view(num_train_images, num_sequences, 4))
        # Obtain bbox prediction
        bbox_pred = self.head(fusion_feat)
        return bbox_pred

    def forward_ref(self, train_imgs, train_bb):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        '''train_imgs: (1,64,3,256,256), test_imgs: (1,64,3,256,256), train_bb: (1,64,4)'''
        num_sequences = train_imgs.shape[-4] # 64
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat与test_feat的数据类型都是OrderedDict,字典的键为'layer2','layer3'等等'''
        train_feat_dict = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))   # 输入size是(64,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        ''' layer2的特征size为(64,512,32,32),  stride=8
            layer3的特征size为(64,1024,16,16), stride=16(我想先只用layer3)'''
        train_feat_list = [feat for feat in train_feat_dict.values()]

        # get reference feature
        self.neck.get_ref_kernel(train_feat_list, train_bb.view(num_train_images, num_sequences, 4))


    def forward_test(self, test_imgs):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        '''train_imgs: (1,64,3,256,256), test_imgs: (1,64,3,256,256), train_bb: (1,64,4)'''
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1 # 1

        # Extract backbone features
        '''train_feat与test_feat的数据类型都是OrderedDict,字典的键为'layer2','layer3'等等'''
        test_feat_dict = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))      # 输入size是(64,3,256,256)
        '''list,其中每个元素对应一层输出的特征(tensor)'''
        ''' layer2的特征size为(64,512,32,32),  stride=8
            layer3的特征size为(64,1024,16,16), stride=16(我想先只用layer3)'''
        test_feat_list = [feat for feat in test_feat_dict.values()]

        # fuse feature from two branches
        fusion_feat = self.neck.fuse_feat(test_feat_list)
        # Obtain bbox prediction
        bbox_pred = self.head(fusion_feat)
        return bbox_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.used_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def SEbb_resnet50(backbone_pretrained=True,used_layers=['layer3'],pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # head
    head_net = bbox.BBox_Predictor(inplanes=pool_size*pool_size)

    net = SEbbnet(feature_extractor=backbone_net, neck_module=neck_net, head_module=head_net,
                  used_layers=used_layers, extractor_grad=False,unfreeze_layer3=unfreeze_layer3)

    return net
@model_constructor
def SEbb_resnet50_anchor(backbone_pretrained=True,used_layers=['layer3'],pool_size=None,unfreeze_layer3=False):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)
    # neck
    neck_net = CorrNL.CorrNL(pool_size=pool_size)
    # head
    '''BBox_predictor_anchor'''
    head_net = bbox.BBox_Predictor_anchor(inplanes=pool_size*pool_size)

    net = SEbbnet(feature_extractor=backbone_net, neck_module=neck_net, head_module=head_net,
                  used_layers=used_layers, extractor_grad=False,unfreeze_layer3=unfreeze_layer3)

    return net
'''maybe  we can try ResNext, SENet... in the future'''