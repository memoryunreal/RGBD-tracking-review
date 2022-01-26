# Complementary Local-Global Search for RGBD Visual Tracking (CLGS_D)

### Hardware
- CPU: Intel Core i9-9900k CPU @ 3.60GHz
- GPU: NVIDIA GeForce GTX 2080 Ti
- Mem: 32G

### Versions
- Ubuntu 18.04.2 LTS
- CUDA 10.0
- Python 3.7

### Usage
**1.**
```bash
conda create -n vot20 python=3.7
conda activate vot20
cd <path/to/CLGS_D>
sh install.sh

pip install yacs
pip install nms
pip install progress
conda install -y scipy=1.1.0

cd ./centernet/lib/models/networks/DCNv2/
./make.sh
cd -
```

If you have any question about installation, please reference [pysot](https://github.com/STVIR/pysot/blob/master/INSTALL.md), [pytracking](https://github.com/visionml/pytracking), [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) and [centernet](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md) or contact us. ( zhaohj1995@gmail.com / haojie_zhao@mail.dlut.edu.cn )


**2.**
- Modify `project_path` in `config_path.py`.
- Download [models](http://pan.dlut.edu.cn/share?id=dcw35pss3uns) to `./models`.
- CenterNet(resdcn18) is used to generate region proposals in both local search stage and global search stage. For convenience, we have ran the centernet detector on each frame of the RGBD dataset. Please download the detection results [cdet_res.zip](http://pan.dlut.edu.cn/share?id=dcw35pss3uns) and unzip to `./models`.

**3.**

- **NOTE:** In order to reproduce results, we have set random seed in our tracker(Line29~33). However, we found the reproducibility is affected by the toolkit.
If you cannot reproduce the results, we recommend that you can change `VOT_FLAG`(Line209) to `False` and run `tracker.py` directly.
If you have any questions, please contact us. ( zhaohj1995@gmail.com / haojie_zhao@mail.dlut.edu.cn )


### References
```
@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}

@inproceedings{zhou2019objects,
    title={Objects as Points},
    author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
    booktitle={arXiv preprint arXiv:1904.07850},
    year={2019}
}

@InProceedings{rtmdnet,
    author = {Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
    title = {Real-Time MDNet},
    booktitle = {European Conference on Computer Vision (ECCV)},
    month = {Sept},
    year = {2018}
}

@InProceedings{IMKDB17,
    author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
    title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
    booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    month        = "Jul",
    year         = "2017",
    url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
}

@misc{flownet2-pytorch,
    author = {Fitsum Reda and Robert Pottorff and Jon Barker and Bryan Catanzaro},
    title = {flownet2-pytorch: Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/NVIDIA/flownet2-pytorch}}
}

```