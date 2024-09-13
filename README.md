# PPA-CODe: **P**rivacy-**P**reserving **A**utoencoder for **C**ollaborative **O**bject **De**tection
This repository is the official implementation of the paper "[**P**rivacy-**P**reserving **A**utoencoder for **C**ollaborative **O**bject **De**tection](https://ieeexplore.ieee.org/document/10667003)", which is originally forked from the [YOLOv5 repository](https://github.com/ultralytics/yolov5).

## Paper

**Paper**: [arXiv](https://arxiv.org/abs/2402.18864), [IEEE-TIP (Early Access)](https://ieeexplore.ieee.org/document/10667003)

**Bibtex citation**:
```
@ARTICLE{ppa-code,
  author={Azizian, Bardia and Bajić, Ivan V.},
  journal={IEEE Transactions on Image Processing}, 
  title={Privacy-Preserving Autoencoder for Collaborative Object Detection}, 
  year={2024},
  volume={33},
  number={},
  pages={4937-4951},
  doi={10.1109/TIP.2024.3451938}
}
```

## Usage
There are 6 main tasks in this project that can be executed through the provided python codes in the root directory:

`val.py`: To validate the model on the object detection and obtain mAP values and the quality of the Model Inversion Attack (MAI).  
`compress.py`: To validate the whole pipeline including the compression part using VVC  to obtain mAP and bitrate values.  
`detect.py`: To run the detection on a source of images or videos and create the bounding boxes on the inputs.  
`reconstruct.py`: To recover the input using a pre-trained adversary (InvNet).  
`train.py`: To train a model without an adversary only with object detection and compressibility loss.  
`train_adversarial.py`: To adversarially train a model with an adversary (RecNet) with object detection, compressibility, and reconstruction loss.  
`train_invnet.py`: To train only the adversary (InvNet) of a model to recover the input using reconstruction loss.  

### Validation
Use `val.py` to run the object detection validation on the whole pipeline (with or without the autoencoder) and obtain the mAP values. If an adversary is available in the weight file, it also computes the quality of the Model Inversion Attack (MIA) through PSNR and MS-SSIM. You can also add 4 different types of noise either to the original latent space, bottleneck of the autoencoder, or input. You can use `val.py` to reproduce the results of Fig. 8a in the paper.
```
python val.py --data data/coco.yaml --weights <path/to/the/weight.pt>
```

### Validation with Compression
`compress.py` is exactly the same as `val.py` but with VVC compression included. It can be used to reproduce the results of Fig. 9a and Fig. 9b in the paper.

*To be able to run `compress.py`, you need to download [VVenC](https://github.com/fraunhoferhhi/vvenc/releases/tag/v1.1.0), build it, and put the executable file *vvencFFapp* into the vvc folder.

* For Benchmark-input, we used `$QP` values of {28, 32, 34, 36, 39, 42, 44}:
```
python compress.py --data data/coco.yaml --qp $QP --compression input
```

* For Benchmark-latent, we used `$QP` values of {39, 41, 42, 43, 44, 45}:
```
python compress.py --data data/coco.yaml --qp $QP --compression bottleneck --chs-in-w 16 --chs-in-h 12 --tensors-min -0.2786 --tensors-max 3.0
```

* For Benchmark-bottleneck, we used `$QP` values of {34, 36, 37, 38, 39, 40}:
```
python compress.py --data data/coco.yaml --weights <path/to/cmprs0_rec0.0.pt> --qp $QP --compression bottleneck --chs-in-w 8 --chs-in-h 8
```

* For Proposed, we used the combinitaion of the following `$QP`s and `$WEIGHT`s to generate the 9 points in the rate-accuracy and privacy-accuracy planes.  
(`$QP`, `$WEIGHT`) =  
(25, [cmprs1_rec1.0.pt](https://drive.google.com/file/d/1wO-n56LUmEeuZkhBF6Il1E0j70Dqupfm/view?usp=sharing)),  
(28, [cmprs1_rec1.0.pt](https://drive.google.com/file/d/1wO-n56LUmEeuZkhBF6Il1E0j70Dqupfm/view?usp=sharing)),  
(31, [cmprs1_rec1.5.pt](https://drive.google.com/file/d/1rz5JhyZ2KYQ9bZzmH2mzYZ4wNnE0dUpD/view?usp=sharing)),  
(31, [cmprs2_rec1.5.pt](https://drive.google.com/file/d/1FbZ3BaWB9sahInSUy1H6nt24B2ktRW-F/view?usp=sharing)),  
(30, [cmprs4_rec2.0.pt](https://drive.google.com/file/d/1_CrfrBt6bOH7c0U_17a5cfAVjMIE7n3y/view?usp=sharing)),  
(33, [cmprs4_rec2.0.pt](https://drive.google.com/file/d/1_CrfrBt6bOH7c0U_17a5cfAVjMIE7n3y/view?usp=sharing)),  
(35, [cmprs4_rec2.0.pt](https://drive.google.com/file/d/1_CrfrBt6bOH7c0U_17a5cfAVjMIE7n3y/view?usp=sharing)),  
(37, [cmprs4_rec2.0.pt](https://drive.google.com/file/d/1_CrfrBt6bOH7c0U_17a5cfAVjMIE7n3y/view?usp=sharing)),  
(37, [cmprs4_rec1.0.pt](https://drive.google.com/file/d/1Cof2lGhNVZwpW0oKPN4jCYO_W4Xc3pc0/view?usp=sharing))
```
python compress.py --data data/coco.yaml --weights <path/to/$WEIGHT> --qp $QP --compression bottleneck --chs-in-w 8 --chs-in-h 8
```

### Detection
Use `detect.py` to run object detection either on an image, a group of images (folder or glob), a URL containing an image, or the webcam stream. It annotates the images with the bounding boxes, class id, and the confidence score. You can also add 4 different types of noise either to the original latent space, bottleneck of the autoencoder, or input. 
```
python detect.py --source <path/to/the/image(s)>
```

### MIA Reconstruction
Use `reconstruct.py` to run Model Inversion Attack (MIA) on the intercepted features and recover the input. It gets the weight file and automatically considers RecNet wights as the adversary (InvNet). Therefore, the InvNet model should be pre-trained for input reconstruction after finishing the adversarial training. All the provided weights in the Model Zoo have already been trained for input reconstruction. You can also add 4 different types of noise either to the original latent space, bottleneck of the autoencoder, or input. You should use `reconstruct.py` to obtain the recovered images from [LFW](https://vis-www.cs.umass.edu/lfw/) and [CCPD](https://github.com/detectRecog/CCPD) datasets to be able to get the face and license plate recognition accuracies as outlined in the paper. It can accept either a single image, a folder of images, or a pattern matching glob.
```
python reconstruct.py --sample-img <path/to/the/image(s)>
```

### Training
Use `train.py` to train a model (with or without an autoencoder) in a typical manner using object detection and compressibility loss. This can be used for the pretraining phase before starting the main adversarial training stage.

* Training the YOLOv5m model from scratch without an autoencoder:
```
python train.py --weights scratch --cfg models/yolov5m.yaml --data data/coco.yaml --train-yolo all
```

* Training the pretrained YOLOv5m model with a randomly-initialized autoencoder:
```
python train.py --data data/coco.yaml --train-yolo nothing --autoenc-chs 192 128 64
```

### Adversarial Training
Use `train_adversarial.py` to train the whole model in an adversarial manner with the auxiliary RecNet model using object detection, compressibility, and reconstruction loss.

```
python train_adversarial.py --weights <path/to/the/pretrained/model.pt> --data data/coco.yaml --w-compress <desired w_cmprs> --w-rec <desired w_rec> --w-grad <desired \beta> --sample-img <path/to/a/sample/image>
```

### InvNet Training
in progress ...


## Model Zoo

| Model Weight | Explanation | w_cmprs | w_rec | \beta | mAP@.5:.95 | MIA PSNR (dB) |
| --- | --- | --- | --- | --- | --- | --- |
| [pretraining](https://drive.google.com/file/d/1G_rU36wgA3jNbWjnmzxAB0giQpDtzaMm/view?usp=sharing) | The pretrained autoencoder with associated trained InvNet (initialization phase) | 0 | N/A | N/A | 43.4 | 21.9 |
| [Benchmark-latent](https://drive.google.com/file/d/1FN7eBTXo0g-YtjZr2lhhgqk8xbiUOYpT/view?usp=sharing) | The *Benchmark-latent* model with associated trained InvNet | 0 | 0 | N/A | 43.7 | 20.3 |
| [Benchmark-bottleneck](https://drive.google.com/file/d/1zq3WIvI4N2YE8yBN9xiHtRMwLsa4aLlw/view?usp=sharing) | The *Benchmark-bottleneck* model with associated trained InvNet | 0 | 0 | N/A | 42.9 | 21.9 |
| [cmprs0_rec0.5](https://drive.google.com/file/d/1LVMSAaC3j4Arby_YOvZrMXHaKn4dssSb/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 0.5 | 5 | 42.8 | 18.7 |
| [cmprs0_rec0.8](https://drive.google.com/file/d/1emLCcqoVJqJqlXKqJWr3CiahER_7Qh4d/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 0.8 | 5 | 42.7 | 17.8 |
| [cmprs0_rec1.2](https://drive.google.com/file/d/1kC_P4f3Nf53dxrBVeU0NiYJC9QBXcFBm/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 1.2 | 5 | 42.4 | 17.7 |
| [cmprs0_rec2.0](https://drive.google.com/file/d/1BM7Rf_Guci_gG-UssdHxrRdOisSEjtB7/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 2.0 | 5 | 42.2 | 17.6 |
| [cmprs1_rec1.0](https://drive.google.com/file/d/1wO-n56LUmEeuZkhBF6Il1E0j70Dqupfm/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 1 | 1.0 | 5 | 42.1 | 17.5 |
| [cmprs1_rec1.5](https://drive.google.com/file/d/1rz5JhyZ2KYQ9bZzmH2mzYZ4wNnE0dUpD/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 1 | 1.5 | 5 | 41.7 | 17.7 |
| [cmprs2_rec1.5](https://drive.google.com/file/d/1FbZ3BaWB9sahInSUy1H6nt24B2ktRW-F/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 2 | 1.5 | 5 | 41.2 | 17.9 |
| [cmprs4_rec1.0](https://drive.google.com/file/d/1Cof2lGhNVZwpW0oKPN4jCYO_W4Xc3pc0/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 4 | 1.0 | 5 | 40.4 | 17.8 |
| [cmprs4_rec2.0](https://drive.google.com/file/d/1_CrfrBt6bOH7c0U_17a5cfAVjMIE7n3y/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 4 | 2.0 | 5 | 40.7 | 17.7 |
| [cmprs0_rec0.1](https://drive.google.com/file/d/15xpJZle6GfatgZG23se7lEY3NteaKLEl/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 0.1 | 0 | 42.9 | 21.2 |
| [cmprs0_rec0.5](https://drive.google.com/file/d/1VxHcMAOomCDuEz3hd7JPwBZJeMtSIKd1/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 0.5 | 0 | 42.9 | 19.9 |
| [cmprs0_rec2.0](https://drive.google.com/file/d/1QK0PlS-07yG_elXKOV5f4mwCIy24sSbx/view?usp=sharing) | The adversarially trained model with associated trained InvNet | 0 | 2.0 | 0 | 42.7 | 19.9 |

