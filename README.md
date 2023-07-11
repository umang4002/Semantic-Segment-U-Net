# 1)Dataset and paper
   
  dataset:https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge
  paper: https://arxiv.org/abs/1505.04597

# 2)Data Preprocessing
  **2.1 Loading images and masks from their directories**
  
    In this step, we will:
    1. Create 2 separate lists containing the paths of images and masks.
    2. Split the lists into training, testing, and validation datasets

    
  **2.2 Creating a function for reading images and masks and returning equivalent arrays**
  
    The *read_image* function will
    1. Read an image and its mask from the paths
    2. Convert the masks and images into arrays
    3. Normalize the datasets
# 3)Model architecture and training

    Semantic segmentation is a computer vision task that involves dividing an image into    
    different regions or segments, where each segment represents a specific object or class 
    of objects. In other words, it assigns a semantic label to each pixel in the image, 
    effectively creating a pixel-level understanding of the scene.
    
    We will use the **U-Net**architecture for our semantic segmentation task. The U-net model
    has a U-shaped architecture. It was created previously in 2015 for Biomedical image      
    segmentation but now has a number of different use cases.
    
    U-Net is an advanced architecture that builds upon the Fully Convolutional Network (FCN) 
    approach. FCN replaces the traditional dense layers in a regular CNN with special layers    
    called transposed convolutions. These transposed convolutions help restore the original 
    size of the input image while preserving important spatial details, which are vital for 
    tasks like image segmentation.

    **U-Net architecture**:
    
    **1. Contracting path(Encoder for downsampling process)**
    
        The contracting path of U-Net follows a typical CNN structure, consisting of 
        convolutional layers, activation functions (ReLU), and pooling layers. These 
        layers work together to reduce the size of the image and extract its important 
        features.
        In detail, it used repeated two 3x3 unpadded convolutions, each followed by a 
        ReLU(Rectified Linear Unit), and a 2x2 max pooling operation with a stride = 2 
        for downsampling. At each downsampling step, the number of feature channels 
        is doubled.
        During the contracting process, convolution outputs are stored in separate variables
        before size reduction. This is passed to the expanding blocks during the upsampling 
        process using **skip layers**.

    **2. Expanding path(Decoder for upsampling)**
    
         Upsampling is performed for bringing back the reduced image from downsampling 
         to its original size while shrinking the channels gradually.
         Every step in the expansive path consists of an upsampling of the feature map 
         followed by a 2x2 convolution (“up-convolution” or "transpose convolution) that 
         halves the number of feature channels while growing the height and width of 
         the image.

    **3.1 U-Net Model Design**
    
          A) Define a function that represents an encoding block in the U-Net model. The     
            function will return the next layer output and the skip connection output for 
            the corresponding block in the model
            
          B) Define a function that represents a decoding block in the U-Net model. This   
             function will merge the skip-connection input with the previous layer, 
             process it, and return an output.
             
          C) Develop a model using both the encoding and decoding blocks output.

        Next is a concatenation with the correspondingly cropped feature map from the   
        downsampling and two 3x3 convolutions followed by ReLU.

    **3.2 Final Feature Mapping Block**
    
        At the final layer, a 1x1 convolution is used to map each 64-component feature 
        vector to the desired number of classes. In total, the network has **23 
        convolutional layers**. The channel dimension from the previous layer corresponds
        to the number of filters used, so when you use 1x1 convolutions, you can 
        transform that dimension by choosing an appropriate number of 1x1 filters. 
        
        
      ![image](https://github.com/umang4002/Semantic-Segment-U-Net/assets/111570202/5210c58d-5d4c-4be8-82ac-72f6b7f1dc7b)


          


          
         
         
      
    
