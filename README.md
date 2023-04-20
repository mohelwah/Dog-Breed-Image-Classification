
# Dog Breed Image Classification using AWS SageMaker

Welcome to the Image Classification using AWS SageMaker project, which is a part of the AWS Machine Learning Engineer Nanodegree program. In this project, I will be using a pre-trained Resnet50 model from the PyTorch vision library and fine-tuning it with hyperparameter tuning and network re-shaping. Additionally, I will be implementing profiling and debugging with hooks to optimize our model's performance. Finally, I will deploy the model and perform inference. This project will provide me with a hands-on experience of using SageMaker to build and deploy a custom image classification model.

## Project Set Up and Installation
For setting up the project in AWS SageMaker, you can follow the steps below:

1-Open the AWS Management Console and navigate to the SageMaker service.

2- Create a new Notebook instance by clicking on the "Create notebook instance" button.

3- Choose a name for your instance, select an instance type, and choose an IAM role that has the required permissions.

3- Under "Git repositories", select "Clone a public Git repository to this notebook instance" and enter the URL of your project's GitHub repository.

4- Click on "Create notebook instance" to create the instance.

5- Once the instance is running, open JupyterLab by clicking on "Open JupyterLab".

6- In JupyterLab, navigate to the directory where your project's files are located.

7- Open the train_and_deploy.ipynb notebook and follow the instructions to complete the project.

## Dataset
### Overview
The project utilizes the Dog Classification Dataset from Udacity for image classification. The dataset can be obtained by following the [link] (https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) provided here.



### Access
The dataset was uploaded to an S3 bucket in AWS. To access the data, the code in the notebook retrieves the data from the S3 bucket using the sagemaker.s3.S3Downloader module. This module allows for the downloading of the data from the S3 bucket to the notebook instance.

## Hyperparameter Tuning
The image classification task is performed using the ResNet model, which employs the deep Residual Learning Framework to simplify the training process. Two fully connected neural networks are added to the top of the pre-trained model to carry out the classification task, with 133 output nodes.

To optimize the model, AdamW from the torch.optim package is utilized as the optimizer. The following hyperparameters are tuned using the hpo.py script:

Learning rate: 0.01x to 100x
Epsilon (eps): 1e-09 to 1e-08
Weight decay: 0.1x to 10x
Batch size: 64 or 128
The hyperparameter tuning is carried out to find the best combination of hyperparameters that maximizes the model's performance.


## Debugging and Profiling
If the debugging output shows fluctuations and inconsistencies between batches, it may indicate a lack of convergence or overfitting. In such cases, regularization techniques such as dropout or weight decay could be applied to prevent overfitting. Alternatively, the batch shuffling technique can be used to ensure that the training data is presented in a random order for each epoch, preventing the model from memorizing the training data's order. Additionally, trying out different neural network architectures, such as increasing the number of layers or changing the activation functions, could also improve model performance.

### Results
The profiler report can be found ./profiler_repot/profiler-output/profiler-report.html


## Model Deployment
We deployed the model on a "ml.t2.medium" instance type, and to set up and deploy our working endpoint, we used the "endpoint_inference.py" script. For testing purposes, we stored a few test images in the "testImages" folder. We used the following approaches for inference:

1- Using the Predictor Object

