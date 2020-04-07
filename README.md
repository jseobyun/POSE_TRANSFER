Pose transfer (Modified Everybody-dance-now)
======

This code is for video2video pose transfer based on [Everybody dance now](https://arxiv.org/abs/1808.07371).

It follows same 2D skeleton normalization strategy but does not include face-enhancement/temporal smoothing. 

Structures of generator/discriminator are [stacked hour glass](https://arxiv.org/abs/1603.06937) and [pix2pix](https://arxiv.org/pdf/1711.11585.pdf).

[Openpose pytorch-reproduced](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) is imported in this code with small modification.

Note that, because of GAN training issue between generator and discriminator, the result shows blurred person. 

This issuse will be fixed.
   


## Preparation
Download the [pretrained-pose model](https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0)

    data
    |-- source
        |-- annotations
        |-- images
        |-- train.mp4 (video of person who you want to make dance)
        |-- test.mp4 (video of dance) 
        
    pose_detector
    |-- weights
        |-- pose_model.pth           
        

## Dependency

[2D pose detector C++ postprocessing](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/tree/master/lib/pafprocess)

[Pytorch]

[OpenCV]


## Usage

#### Step1 : 2D skeleton extraction
Run data/video2data.py with mode 'train' and mode 'test'

It makes training images and 2D training/testing labels on data/images and data/annotations

#### Step2 : 2D skeleton normalization

Run data/annot_normalize.py

It creates a normalized 2D label considering scale and position at data/annotations. 

Note that the normalization highly depends on the videos because it only uses 2D information.

It can fail when people in train/test videos have significantly different scale or train/test videos include a dance which changes 2D skeleton extremely compared to standing position.

#### Step3 : Training

Run main/train.py

It saves the model per epoch and you can continue training by changing main/config.py 

#### Step4 : Testing

Run main/test.py, and then output/img2video.py

It saves the result video at output/result

      

 



 