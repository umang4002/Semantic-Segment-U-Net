## **1. Dataset and paper**
   
  dataset:https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge
  
  paper: https://arxiv.org/abs/1505.04597

## **2. Data Preprocessing**
<a class="anchor" id="2-1" name="2-1"></a>

### **2.1 Loading images and masks from their directories**
  
    In this step, we will:
    1. Create 2 separate lists containing the paths of images and masks.
    2. Split the lists into training, testing, and validation datasets

    
#### **2.2 Creating a function for reading images and masks and returning equivalent arrays**
  
    The read_image function will
    
    1. Read an image and its mask from the paths
    
    2. Convert the masks and images into arrays
    
    3. Normalize the datasets
    
## **3. Model architecture and training**

    Semantic segmentation is a computer vision task that involves dividing an image into    
    different regions or segments, where each segment represents a specific object or class 
    of objects. In other words, it assigns a semantic label to each pixel in the image, 
    effectively creating a pixel-level understanding of the scene.
    
    We will use the U-Net architecture for our semantic segmentation task. The U-net model
    has a U-shaped architecture. It was created previously in 2015 for Biomedical image      
    segmentation but now has a number of different use cases.
    
    U-Net is an advanced architecture that builds upon the Fully Convolutional Network (FCN) 
    approach. FCN replaces the traditional dense layers in a regular CNN with special layers    
    called transposed convolutions. These transposed convolutions help restore the original 
    size of the input image while preserving important spatial details, which are vital for 
    tasks like image segmentation.

 ### U-Net architecture:
     
   **1 .Contracting path(Encoder for downsampling process)**
    
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
        process using skip layers.

   **2. Expanding path(Decoder for upsampling)**
    
         Upsampling is performed for bringing back the reduced image from downsampling 
         to its original size while shrinking the channels gradually.
         Every step in the expansive path consists of an upsampling of the feature map 
         followed by a 2x2 convolution (“up-convolution” or "transpose convolution) that 
         halves the number of feature channels while growing the height and width of 
         the image.

  ### **3.1 U-Net Model Design**
    
          A) Define a function that represents an encoding block in the U-Net model. The     
            function will return the next layer output and the skip connection output for 
            the corresponding block in the model
            
          B) Define a function that represents a decoding block in the U-Net model. This   
             function will merge the skip-connection input with the previous layer, 
             process it, and return an output.
             
          C) Develop a model using both the encoding and decoding blocks output.

        Next is a concatenation with the correspondingly cropped feature map from the   
        downsampling and two 3x3 convolutions followed by ReLU.

   **3. Final Feature Mapping Block**
    
        At the final layer, a 1x1 convolution is used to map each 64-component feature 
        vector to the desired number of classes.  The channel dimension from the previous 
        layer corresponds to the number of filters used, so when you use 1x1 convolutions, 
        you can transform that dimension by choosing an appropriate number of 1x1 filters. 
        
   In total, the network has 23 convolutional layers.
   ![image](https://github.com/umang4002/Semantic-Segment-U-Net/assets/111570202/ac392078-b89c-4bc4-abaf-f95e267c3777)

 ## **4. Model Evaluation**
     
     Model Evaluation is a critical step in the development process to determine the 
     effectiveness of our model and how well it will perform in the future. When dealing with       classification tasks, relying solely on model accuracy may not provide a complete picture       of its performance, especially when dealing with imbalanced datasets. In dense                prediction tasks like image segmentation, where the goal is to simplify or change the          representation of an image into meaningful classes, it becomes even more challenging to        assess the model's ability to accurately partition different classes.

     To overcome these limitations, we employ additional metrics such as precision, recall,         Intersection over Union (IoU), and F1-score to evaluate our model's performance. These         metrics provide valuable insights into how well our model performs in partitioning             different classes. By calculating the confusion matrix between the predicted             
     segmentations and the ground truth segmentations, we can identify true positives (TP), 
     true negatives (TN), false positives (FP), and false negatives (FN). Using these 
     values, we compute metrics like recall, precision, IoU, and F1-score to assess the 
     model's performance.

      In summary, model evaluation goes beyond simple accuracy measurements when dealing with 
      dense prediction tasks like image segmentation. By incorporating metrics like recall, 
      precision, IoU, and F1-score, we gain a more comprehensive understanding of how well our 
      model performs in accurately partitioning different classes within an image.

      The expressions for these metrics are defined as:
      
         1) Precision = TP/(TP + FP)
         
         2) Recall/Sensitivity = TP/(TP + FN)
         
         3) Intersection over Union (IoU)/Jaccard Similarity = TP/(TP + FP + FN)
         
         4)F1-score(JS)/Dice coefficient = 2 * ((Precision * Recall)/(Precision + Recall))
         

      To carry out these evaluations, we will:

      1)Create segmentations/masks of images in our dataset
      
      2)Evaluate predicted segmentations
      
  ## **5. Predict image segmentations using the trained model**

     Though our model is performing well, visualizing it how it performs on these datasets
     could give us additional gains.

     1)It invloves creating a function for preprocessing selected images and display their 
       true mask and predicted mask.
       
     2) Predict and compare masks of images in the training set.

     3) Predict and compare masks of image in the validation set.

     4) Predict and compare masks of images in the test set.

## **6. Segmented image**
![image](https://user-images.githubusercontent.com/84759422/177004225-256b1ae9-b31e-47e5-bd8d-6ca5216d70cf.png)

#### Predicted mask shows the segmented image that our model has predicted.
     
     

      

        
         

          


          
         
         
      
    
