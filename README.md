# Ear Landmark Detection with CNN

There are 55 landmarks on human ear which help identifying the person. This model has 500 training images, 105 test images and corresponding landmarks focused on right ear. I also modified the images and landmarks to represent other forms such as left ear, rotated etc. You can get 3000 training 630 test images with this method if you run CreateDataSet.py. Before that, download the data folder from the link below. You can see names of the files & folders are compatible with the functions. 

Original data belongs to this [source](https://ibug.doc.ic.ac.uk/resources/ibug-ears/) (check out Collection A).
I renamed and regrouped the content to ease preprocessing of the model. The data that is compatible with the model can be accessed from [here](https://www.dropbox.com/sh/c8hizptl60lfogh/AADQN-kkuzkiP3ZcREQRxERsa?dl=0).

This model was built from scratch. It takes 224x224 images as input and outputs the predicted landmark points. You can see the examples below. 

### Input
![right ear](/images/test_11.png)

### Output
![right ear w/landmarks](/images/result_11.png)

### Input
![left ear](/images/test_198.png)

### Output
![left ear w/landmarks](/images/result_198.png)

## Architecture
![model architecture](/images/modelarch.jpg)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkbulutozler%2Fear-landmark-detection-with-CNN&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
