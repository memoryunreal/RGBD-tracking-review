# TIP
## Install the environment
- Use the Anaconda
```
conda create -n tip python=3.7
conda activate tip
bash TIP/install.sh

# DAL
conda create -n dal python=3.7
conda activate dal
bash TIP/install_CLGSD.sh
```
## VOT configuration
> ### Accelete speed when running multiple vot evaluate commands
- Find VOT Packages vot.py
```
anaconda3/envs/to/site-packages/vot/dataset/vot.py line 120

import random
random.shuffle(names) # evaluate from random initialized sequences avoid the trackers accessing to the same sequence
```
> ### Support args for your running script
- Find VOT Packages trax.py
```
anaconda3/envs/to/site-packages/vot/tracker/trax.py line 464

# simple check if the command is only a package name to be imported or a script
# if re.match("^[a-zA-Z_][a-zA-Z0-9_\\.]*$", command) is None:
#     # We have to escape all double quotes
#     command = command.replace("\"", "\\\"")
#     command = '{} -c "import sys;{} {}"'.format(interpreter, pathimport, command)
# else:
#     command = '{} -m {}'.format(interpreter, command)
command = '{} -m {}'.format(interpreter, command) # using this command will support adding args in trackers.ini
```
- Edit trackers.ini
```
/workspace/trackers.ini

[TSDM]
label = TSDM
protocol = traxpython

command = vot_DSiamRPN --gpu 2 # here we can add some args for the vot_DSiamRPN script to capture.

paths = /home/dataset/TIP/TSDM/TSDM/Evaluation_VOT

```

> ### DeT && DAL
```
# DeT run command (stark_tip)
python /home/dataset/TIP/DeT/pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name all --input_dtype rgbcolormap --gpuid 1 --resultpath /home/dataset/TIP/workspace/synchronize1/ --asynchronize 1

# DAL run comand (CLGSD)
python /home/dataset/TIP/DAL/pytracking_dimp/pytracking/run_tracker.py dimp_rgbd_blend dimp50_votd --dataset alldata --gpuid 6 --resultpath /home/dataset/TIP/workspace/synchronize1/ --asynchronize 1
```

> ### Asynchromize VOT dataset
- Edit config.yaml
```
# asynchronize frame 1/2/5/10
registry:
- ./trackers.ini
stack: votrgbd2019
asynchronize: 1
```
- Find VOT Packages
```
anaconda3/envs/to/site-packages/vot/workspace/__init__.py line 148

'''
    asynchronize
'''
try:
    asynchronize = int(kwargs['asynchronize'])
    kwargs.pop('asynchronize')
    print("asynchronize, frame: {}".format(asynchronize))
except:
    asynchronize = 0
    print("no asynchronize")
'''
    asynchronize
'''
self._directory = directory
self._storage = LocalStorage(directory) if directory is not None else NullStorage()

super().__init__(**kwargs)
dataset_directory = normalize_path(self.sequences, directory)

if not self.stack.dataset is None:
    Workspace.download_dataset(self.stack.dataset, dataset_directory)
'''
    asynchronize
'''
self._dataset = load_dataset(dataset_directory, asynchronize=asynchronize )
'''
    asynchronize
'''
```

```
anaconda3/envs/to/site-packages/vot/dataset/__init__.py line 557

def load_dataset(path: str, asynchronize=0) -> Dataset:
    """Loads a dataset from a local directory

    Args:
        path (str): The path to the local dataset data

    Raises:
        DatasetException: When a folder does not exist or the format is not recognized.

    Returns:
        Dataset: Dataset object
    """

    if not os.path.isdir(path):
        raise DatasetException("Dataset directory does not exist")

    if VOTDataset.check(path):
        return VOTDataset(path, asynchronize=asynchronize)
    elif GOT10kDataset.check(path):
        return GOT10kDataset(path)
    elif OTBDataset.check(path):
        return OTBDataset(path)
    elif TrackingNetDataset.check(path):
        return TrackingNetDataset(path)
    else:
        raise DatasetException("Unsupported dataset type")
```

```
anaconda3/envs/to/site-packages/vot/dataset/vot.py line 25 & 111 
# line 25
def __init__(self, base, name=None, dataset=None, asynchronize=0):
    self._base = base
    '''
    synchronize
    '''
    self.asynchronize = asynchronize
    '''
    synchronize
    '''
    if name is None:
        name = os.path.basename(base)
    super().__init__(name, dataset)
    
# line 111
    for name, value in values.items():
        if not len(value) == len(groundtruth):
            raise DatasetException("Length mismatch for value %s" % name)
    '''
        asynchronize
    '''
    if not self.asynchronize == 0:
        channels["depth"]._files = channels["depth"]._files[self.asynchronize:]
        for syn in range(self.asynchronize):
            channels["depth"]._files.append(channels["depth"]._files[-1])
    '''
        asynchronize
    '''
    return channels, groundtruth, tags, values

# line 123
class VOTDataset(Dataset):

    def __init__(self, path, asynchronize=0):
        self.asynchronize = asynchronize # pass args to votsequence
        super().__init__(path)

        if not os.path.isfile(os.path.join(path, "list.txt")):
            raise DatasetException("Dataset not available locally")

        with open(os.path.join(path, "list.txt"), 'r') as fd:
            names = fd.readlines()

        self._sequences = OrderedDict()

        with Progress("Loading dataset", len(names)) as progress:

            for name in names:
                self._sequences[name.strip()] = VOTSequence(os.path.join(path, name.strip()), dataset=self, asynchronize=self.asynchronize)
                progress.relative(1)
```
# Robustness 
- Configuration
> Asynchronize
```
Host: 10.20.111.5
Path: /home/chenhongjun/lz/vot2022
Run docker: /home/chenhongjun/lz/TIP2022/tip2022.sh
Dockerimage: watchtowerss/vot2022:latest
Workspace: /disk2/tracking/TIP
annoconda envs: stark_tip for iiau,DeT,TSDM CLGSD for DAL
```
- DeT
```
python /home/dataset/TIP/DeT/pytracking/run_tracker.py dimp DeT_DiMP50_Max --dataset_name all --input_dtype rgbcolormap --gpuid 1 --resultpath /home/dataset/TIP/workspace/synchronize1/ --asynchronize 1
```
- DAL
```
python /home/dataset/TIP/DAL/pytracking_dimp/pytracking/run_tracker.py dimp_rgbd_blend dimp50_votd --dataset alldata --gpuid 6 --resultpath /home/dataset/TIP/workspace/synchronize1/ --asynchronize 1
```
- TSDM & iiau_rgbd
```
vot evaluate --workspace ./ TSDM iiau_rgbd
```

## Model path
* Models and results for mixformer 
[[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver)  [[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv) 
* Download the pre-trained  [Alpha-Refine](https://drive.google.com/open?id=1qOQRfaRMbQ2nmgX1NFjoQHfXOAn609QM)  network   (vot only)

```
Download models from Google Driver or Baidu Driver 
Models path: /Project/to/MixFormer/lib/test/networks/mixformer*.pth.tar

# Download Alpha-Refine models (vot only)
Model path: /external/AR/ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384/ARnet_seg_mask_ep0040.pth.tar
```

## Experiments on VOT2022 RGBD
- Before evaluating on VOT2022 RGBD, please install some extra packages following [external/AR/README.md](external/AR/README.md). Also, the VOT toolkit is required to evaluate our tracker. To download and instal VOT toolkit, you can follow this [tutorial](https://www.votchallenge.net/howto/tutorial_python.html). For convenience, you can use our example workspaces of VOT toolkit under ```external/VOT2022RGBD/``` by settings ```trackers.ini```.

```
# Check the tracker.ini
vim external/VOT2022RGBD/tracker.ini

# Create vot workspace dir
vot initialize vot2022/rgbd --workspace ./

# Evaluate
vot evaluate --workspace ./ ProMixTrack

# Analysis
vot analysis --workspace ./ ProMixTrack
```



## Contact
- Zhe Li: liz8@mail.sustech.edu.cn

## Model Zoo and raw results
- The trained models and the raw tracking results are provided in the [[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver) or
[[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv).

## Acknowledgments
* Thanks for [VOT](https://www.votchallenge.net/) Library and [Mixformer](https://github.com/MCG-NJU/MixFormer) Library, which helps us to quickly implement our ideas.
