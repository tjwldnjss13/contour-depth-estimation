# Monocular Depth Estimation using Attention Module and Contour Prediction

Method to improve monocular depth estimation accurcy using CBAM and contour prediction module.

## CDNet (Contour Depth Network)
CDNet is U-Net shaped network with CBAM and contour prediction.

### Model
#### Encoder
![인코더구조](https://user-images.githubusercontent.com/48514976/158320497-982e4da1-8763-4251-b583-ce049d7f1e01.JPG)


#### Decoder
![디코더 구조](https://user-images.githubusercontent.com/48514976/158320504-8025332f-d3e5-4a16-9635-0afc7747881a.PNG)


## Result
### Evaluation
  
Evaluation using KITTI rader depth ground truth.
  
#### Error

  Network with contour prediction has better results perspective of Sq Rel and RMSE error compared to the one without contour prediction.

||Abs Rel|Sq Rel|RMSE|RMSE log|
|-|-|-|-|-|
|Without contour|0.101|0.956|4.902|0.175|
|With contour|0.102|0.905|4.843|0.175|

#### Accuracy

  Both networks has similar results in perspective of accuracy.
  
||delta < 1.25|delta < 1.25^2|delta < 1.25^3|
|-|-|-|-|
|Without contour|0.896|0.964|0.984|
|With contour|0.895|0.966|0.985|

### Visualization
  
![3_image_left](https://user-images.githubusercontent.com/48514976/158320685-e92b9d4f-adef-45b2-9509-ef67f84c67cf.png)
![cdnet5_3_image_left](https://user-images.githubusercontent.com/48514976/158320692-062532ea-4535-4484-879e-9909329353b0.JPG)

