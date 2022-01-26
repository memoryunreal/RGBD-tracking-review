# TSDM
A RGB-D tracker base on CNN with using depth information.  
Download paper https://arxiv.org/abs/2005.04063  
The code is simple here.

### Requirment Libs
* Conda with Python 3.7.
* Nvidia GPU.
* PyTorch 1.0
* OpenCV

### Requirment models
* Res20.pth: retrained SiamRPN++ model (works with M-g)
* modelRes: not retrained SiamRPN++ model
* High-Low-two.pth: D-r model  
Pease put the three models into data_and_result/weight. Link as follow:  
BAIDU YUN:    https://pan.baidu.com/s/1Z2c9SymPIRTA_-4p5W1hHA     pin: faon  
Google Drive: https://drive.google.com/drive/folders/17EN9IU-GOhFQt7middHVaNQFwWj7U8MP?usp=sharing  



### Requirment datasets
* VOT-RGBD2019: www.votchallenge.net/
* PTB dataset:  http://tracking.cs.princeton.edu/dataset.html
* demo dataset: https://drive.google.com/drive/folders/19O6o8H_CblZehAJpNQ_KIFe2XGVMs_eY?usp=sharing  
Please put three datasets above into "data_and_result/test_data/". If you only run demo, you can only download demo dataset and put it into "data_and_result/test_data/"


### Test demo
If you wang to run the demo, please run "python3 test.py --sequence n". Here, n ranges {1,2,3}.

### Acknowlege
Thanks to SiamRPN++, it is the core of our model. Paper in https://arxiv.org/pdf/1812.11703.pdf

### Others
I will always update and maintain this repository.
