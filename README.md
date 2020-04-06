POSE TRANSFER (modified Everybody-dance-now)
======
Generator structure : Stacked hour glass

Discriminator structure : Multi scale discriminator from pix2pix

This code generates little bit blurred output because of GAN training difficulty. 

## Usage

#####Step1 : 2D skeleton extraction
Put train.mp4(video with person who you want to make dance) and test.mp4(video with the dance) to data/source

Run data/video2data.py with mode 'train' and mode 'test'

It makes training images and 2D labels on data/images and data/annotations

#####Step2 : 2D skeleton normalization

Run data/annot_normalize.py

It creates a normalized 2D label considering scale and position at data/annotations. 

Note that the normalization highly depends on the datasets because it only uses 2D information.

It can fail when people in train/test video have significantly different scale or train/test video includes a dance which changes 2D skeleton extremely compared to standing position.

#####Step3 : Training

Run main/train.py

It saves the model per epoch and you can continue training by changing main/config.py 

#####Step4 : Testing

Run main/test.py, and then output/img2video.py

It saves the result video at output/result

      

 



 