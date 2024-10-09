# HerdNet  
  
HerdNet is an advanced deep learning model designed for the accurate detection and counting of African mammals in aerial images. This model is introduced in the research paper ["From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning?"](https://www.sciencedirect.com/science/article/pii/S092427162300031X?via%3Dihub) by Alexandre Delplanque and colleagues.  
  
## Model Overview  
  
HerdNet is inspired by CenterNet, which is a neural network based on convolutional layers designed for object detection tasks. The architecture of HerdNet is tailored to handle the challenges of locating and counting dense herds in varied landscapes. It focuses on a localization head from CenterNet for detecting animal centers and includes a classification head for species identification.  
  
## Features  
  
- Optimized for speed vs. accuracy trade-off.  
- Utilizes a modified encoder-decoder structure for efficiency.  
- Employs a Local Maxima Detection Strategy (LMDS) for precise localization during testing.  
  
## Resources  
  
The original code repository and pretrained models are available at:  
[https://github.com/Alexandre-Delplanque/HerdNet.git](https://github.com/Alexandre-Delplanque/HerdNet.git)  
  
## Citation  
  
If you use HerdNet in your research, please cite the original paper by Alexandre Delplanque and his team.  
  
## License  
  
Refer to the repository [link](https://github.com/Alexandre-Delplanque/HerdNet.git) for licensing information.
