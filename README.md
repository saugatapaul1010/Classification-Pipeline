# Machine Learning pipeline for training classification algorithms.

This repository contains an end to end Deep Learning pipeline for training classification models on images using pre trained model in Keras. This pipeline can be used to train any classification algorithms for images provided the data is fed to the pipeline in the correct way.

# Clone the repository:

Open the terminal in your Linux PC in the root folder of this project. Alternatively, you can use Anaconda Navigator shell if you are using Windows.

Please go to this link in order to understand how to clone and use a repo: https://www.linode.com/docs/development/version-control/how-to-install-git-and-clone-a-github-repository/

# Input data directory structure:

The raw data needs to be present inside the 'data' folder located at the root directory of this project. The dataset I have here contains 5 classes of animal images all of which are placed inside the 'data' directory. For simplicity, I have prepared the image dataset using the Flickr API. Please check the iPython notebook at this link: https://github.com/saugatapaul1010/Building-extremely-powerful-object-recognizers-using-very-little-data/blob/master/Transfer%20Learning/01.%20Prepare%20the%20dataset.ipynb. By deafult, this code can download any number of categories and separate them into train, validation and test folders. However, as a pre-requisite to this pipeline, the 'data' folder should contain the category/class folders with the raw images - eg. 'dog', 'frog', 'giraffe', 'horse', 'tiger' and so on. The bottomline is, you can add as many folders as you want as categories and place it inside the data folder and the pipeline will take care of the rest. It will process the data and produce train, validation and test datasets. The train and validation data are used to train the model. The test data is completely unseen by the model, and will be used to report it's performance.

# Installing dependencis:

```
pip install -r requirements.txt
```

Go to the root folder of this project in your shell and execute the above command. This will install all the dependencies that this project needs.

# Downloading the pre-trained model weights:

'''

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5

wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5

'''

Download and place these weights in the '/weights' folder. These are the pre-trained architectures

# Data Preperation:

```
python data_preprocess.py --val_split 0.2 --test_split 0.3
```

```
options:

    --val_split          : fraction of train data to be used as validation data.
                           default=0.2
    --test_split         : fraction of original data to used as test data.
                           default=0.2
    --source             : source directory of the input images.
                           default location is the path of 'df_path'
```

Before executing the above command, the variable 'df_path' should point to the location of the data folder. Change this according the path in your system. The script will take images folders as inputs and create a csv file which will contain all the image names along with the full path names and it's corresponding class label. It will then ranodmly split the CSV file into three seperate files - 'train.csv', 'val.csv' and 'test.csv'. The files will be located inside a 'data_df' folder inside the root directory. This approach allows us to save disk space and computational power and also the time taken to copy-paste the images to three different locations. While splitting the dataset, the 'stratify' option is set to True in order to tackle the problem of imbalanced datasets.

# Training the model

```
python train_pipeline.py --model_name vgg16 --epochs1 10 --epochs2 20 --metric accuracy
```

```
options:

    --model_name         : choose the type of model you want to train with. you can select any one
                           of these: vgg16, inceptionv3, resnet50, inception_resnet, nasnet, xception.
                           default='vgg16'
    --dense_neurons      : enter the number of neurons you want for the pre-final layer. you can select
                           integer number that you want.
                           default=1024
    --batch_size         : enter the number of batches for which the model should be trained on.
                           default=32
    --stage1_lr          : enter the learning rate for stage 1 training. the learning rate for stage 1
                           training will be usual. In stage1, we will freeze the convolution layers and
                           train only the newly added dense layers.
                           default=0.001
    --stage2_lr          : enter the learning rate for stage 2 training. in stage 2, the learning rate has
                           to be kept very low so as to not drastically change the pre-trained weights by
                           massive gradient updates.
                           default=0.000001
    --monitor            : enter the metric you want to monitor. you can enter any metric here that you want
                           the model to monitor, for example 'val_loss' or any other custom metric that you
                           wish to run.
                           default='val_accuracy'
    --metric             : enter the metric you want the model to optimize.
                           default='accuracy'
    --epochs1            : enter the number of epochs you want the model to train for in stage 1.
                           default=10
    --epochs2            : enter the number of epochs you want the model to train for in stage 2. in stage 2,
                           the model needs to train for more epochs than the first stage, since the gradient
                           updates at each epoch will be low due to the low learning rate.
                           default=20
    --finetune           : state 'yes' or 'no' to say whether or not you want to fine tune the convolution block.
                           you can chose 'no', but in that case no stage 2 training will be done and the model
                           parameters won't be fine-tuned to improve it's generalization error on unseen data.
                           default='yes'

```

On executing the above command in your shell, the training will start at both the stages one after the other. Simultaneous evaluation of the model will be done on unseen data as soon as the training gets over in a particular stage. The evaluation results will be saved in the '/evaluation' folder in the root directory of the project. The final report will contain a detailed list of evaluation metrics commonly used in various classification settings exported to an html file, the confusion, precision and recall matrices, the train vs validation loss curves and the classification report. These reports will help us analyze the model performance on test data set.

Prior to the evaluation stages, let's briefly understand what's happening inside 'train_pipeline.py'. With the parameters that the default constructor has already recieved, the training will start for stage 1. In stage 1, we are basically adding two dense layers and connecting them to the output of the final convolution block of the pre-trained architecture. We will chose any pre-trained models for this and train the final layers according to the specific data we have. In order to achieve this, we will freeze all the layers of the convultion base and keep only the last two dense layers as not frozen, so that we can train them without changing the weights of the convolution block. After stage 1 training, in stage 2 we will fine tune the model to further increase it's accuracy. Will keep all the base layers of the convolution block frozen untill some last few layers. So, in stage 2, the top most convolution blocks along with the dense layers will be fine tuned by using a very low learning rate. Using an extremely low learning rate is essential for the second stage so as to ensure that massive gradients updates which may occur due to a high learning lrate wont wreck the weight distribution of the pre-trained architecture. At each stage, the model histories and the best models will be saved in the '/models' folder in the root directory of the project. Please go through the doc strings for any further information about specific functions in the file.

# Model evaluation:

```
python eval_pipeline.py --model_name vgg16 --stage_num 2
```

```
options:

    --model_name         : choose the type of model you want to evaluate with.
                           default='vgg16'
    --stage_num          : enter the name of the stage that you want to evaluate models from.
                           default=2
```

By default, the eval_pipeline script will automatically be executed during and after the course of training. You don't have to manually input the parameters. This is handy just in case you need to generate reports for a particualr model or a particualar stage.

Let's briefly understand what's happening in the evaluation stage. The trained models which are saved in '/models' folder are loaded into the memory and it's evaluated by recording the performance of the model on a completely unseen test data. All reports related to model evaluation is placed inside the '/evaluation' folder in the root directory. At the end of the evaluation phase all these reports will be generated - the train vs loss validation curves to determine and overfitted or underfitted model, the classification report exported to a CSV file, the confusion matrix, precision and recall matrices exported as PNG files, a classification report, a comprehensive report analysis containing all possible classification metrics that are used in general. The complete report is exported into an html file and saved under the '/evaluation' folder. Additionally, there are numerous component CSV files which stores the results for only a certain kind of metric. For further details please read the doc strings for each of the functions.
