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
sudo install graphviz
```

Go to the root folder of this project in your shell and execute the above command. This will install all the dependencies that this project needs. Please note that in Linux, graphviz needs to be installed with sudo and not pip. For configuring graphviz in windows, you can refer to this link: http://graphviz.org/

# Downloading the pre-trained model weights:

```

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/titu1994/Keras-NASNet/releases/download/v1.2/NASNet-large-no-top.h5

wget https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5

```

Download and place these weights in the '/weights' folder. These are the pre-trained architectures that are used in this project - vgg16, inceptionv3, resnet50, inception_resnet, nasnet, xception

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

This is how a sample dataframe would like look after this stage:

<img src='https://github.com/saugatapaul1010/Classification-pipeline-for-transfer-learning/blob/master/images/dataframe_eval.jpg'>

# Training the model:

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

# Reports:

This is how the classification report will look like:

<img src="https://github.com/saugatapaul1010/Classification-pipeline-for-transfer-learning/blob/master/images/classification_report.jpg">

This is how the confusion matrix dataframe looks like:

<img src='https://github.com/saugatapaul1010/Classification-pipeline-for-transfer-learning/blob/master/images/cm_matrix.jpg'>

The below plot will be generated to see the training curve:

<img src='https://github.com/saugatapaul1010/Classification-pipeline-for-transfer-learning/blob/master/evaluation/vgg16_history_stage_1.png'>

Sample precision matrix:

<img src='https://github.com/saugatapaul1010/Classification-pipeline-for-transfer-learning/blob/master/evaluation/vgg16_precision_matrix_stage_1.png'>

<html>
<head>
<title>C:\Users\206255\Desktop\Saugata Paul\Classification-pipeline-for-transfer-learning\evaluation\vgg16_detailed_metrics_analysis_stage_2</title>
</head>
<body>
<h1 style="border-bottom:1px solid black;text-align:center;">PyCM Report</h1><h2>Dataset Type : </h2>
<ul>

<li>Multi-Class Classification</li>

<li>Balanced</li>
</ul>
<p><span style="color:red;">Note 1</span> : Recommended statistics for this type of classification highlighted in <span style="color :aqua;">aqua</span></p>
<p><span style="color:red;">Note 2</span> : The recommender system assumes that the input is the result of classification over the whole data rather than just a part of it.
If the confusion matrix is the result of test data classification, the recommendation is not valid.</p>
<h2>Confusion Matrix : </h2>
<table>
<tr  align="center">
<td>Actual</td>
<td>Predict
<table style="border:1px solid black;border-collapse: collapse;height:42em;width:42em;">
<tr align="center">
<td></td>
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">dog</td>
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">frog</td>
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">gira...</td>
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">horse</td>
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">tiger</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">dog</td>
<td style="background-color:	rgb(206,206,206);color:black;padding:10px;height:7em;width:7em;">6</td>
<td style="background-color:	rgb(206,206,206);color:black;padding:10px;height:7em;width:7em;">6</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
<td style="background-color:	rgb(239,239,239);color:black;padding:10px;height:7em;width:7em;">2</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">frog</td>
<td style="background-color:	rgb(214,214,214);color:black;padding:10px;height:7em;width:7em;">5</td>
<td style="background-color:	rgb(223,223,223);color:black;padding:10px;height:7em;width:7em;">4</td>
<td style="background-color:	rgb(214,214,214);color:black;padding:10px;height:7em;width:7em;">5</td>
<td style="background-color:	rgb(198,198,198);color:black;padding:10px;height:7em;width:7em;">7</td>
<td style="background-color:	rgb(181,181,181);color:black;padding:10px;height:7em;width:7em;">9</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">gira...</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
<td style="background-color:	rgb(214,214,214);color:black;padding:10px;height:7em;width:7em;">5</td>
<td style="background-color:	rgb(206,206,206);color:black;padding:10px;height:7em;width:7em;">6</td>
<td style="background-color:	rgb(231,231,231);color:black;padding:10px;height:7em;width:7em;">3</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">horse</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
<td style="background-color:	rgb(206,206,206);color:black;padding:10px;height:7em;width:7em;">6</td>
<td style="background-color:	rgb(223,223,223);color:black;padding:10px;height:7em;width:7em;">4</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
<td style="background-color:	rgb(223,223,223);color:black;padding:10px;height:7em;width:7em;">4</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:10px;height:7em;width:7em;">tiger</td>
<td style="background-color:	rgb(198,198,198);color:black;padding:10px;height:7em;width:7em;">7</td>
<td style="background-color:	rgb(198,198,198);color:black;padding:10px;height:7em;width:7em;">7</td>
<td style="background-color:	rgb(198,198,198);color:black;padding:10px;height:7em;width:7em;">7</td>
<td style="background-color:	rgb(190,190,190);color:black;padding:10px;height:7em;width:7em;">8</td>
<td style="background-color:	rgb(247,247,247);color:black;padding:10px;height:7em;width:7em;">1</td>
</tr>
</table>
</td>
</tr>
</table>
<h2>Overall Statistics : </h2>
<table style="border:1px solid black;border-collapse: collapse;">
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#95%-CI" style="text-decoration:None;">95% CI</a></td>
<td style="border:1px solid black;padding:4px;">(0.10703,0.22631)</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#ACC_Macro" style="text-decoration:None;">ACC Macro</a></td>
<td style="border:1px solid black;padding:4px;">0.66667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AUNP" style="text-decoration:None;">AUNP</a></td>
<td style="border:1px solid black;padding:4px;">0.47917</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AUNU" style="text-decoration:None;">AUNU</a></td>
<td style="border:1px solid black;padding:4px;">0.47917</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Bennett's-S" style="text-decoration:None;">Bennett S</a></td>
<td style="border:1px solid black;padding:4px;">-0.04167</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#CBA-(Class-balance-accuracy)" style="text-decoration:None;">CBA</a></td>
<td style="border:1px solid black;padding:4px;">0.16196</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#CSI-(Classification-success-index)" style="text-decoration:None;">CSI</a></td>
<td style="border:1px solid black;padding:4px;">-0.66566</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Chi-squared" style="text-decoration:None;">Chi-Squared</a></td>
<td style="border:1px solid black;padding:4px;">17.19048</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Chi-squared-DF" style="text-decoration:None;">Chi-Squared DF</a></td>
<td style="border:1px solid black;padding:4px;">16</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Conditional-entropy" style="text-decoration:None;">Conditional Entropy</a></td>
<td style="border:1px solid black;padding:4px;">2.22184</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Cramer's-V" style="text-decoration:None;">Cramer V</a></td>
<td style="border:1px solid black;padding:4px;">0.16927</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Cross-entropy" style="text-decoration:None;">Cross Entropy</a></td>
<td style="border:1px solid black;padding:4px;">2.32563</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#F1_Macro" style="text-decoration:None;">F1 Macro</a></td>
<td style="border:1px solid black;padding:4px;">0.16693</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#F1_Micro" style="text-decoration:None;">F1 Micro</a></td>
<td style="border:1px solid black;padding:4px;">0.16667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Gwet's-AC1" style="text-decoration:None;">Gwet AC1</a></td>
<td style="border:1px solid black;padding:4px;">-0.04158</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#Hamming-loss" style="text-decoration:None;">Hamming Loss</a></td>
<td style="border:1px solid black;padding:4px;">0.83333</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Joint-entropy" style="text-decoration:None;">Joint Entropy</a></td>
<td style="border:1px solid black;padding:4px;">4.54376</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kullback-Liebler-divergence" style="text-decoration:None;">KL Divergence</a></td>
<td style="border:1px solid black;padding:4px;">0.0037</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kappa" style="text-decoration:None;">Kappa</a></td>
<td style="border:1px solid black;padding:4px;">-0.04167</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kappa-95%-CI" style="text-decoration:None;">Kappa 95% CI</a></td>
<td style="border:1px solid black;padding:4px;">(-0.11622,0.03288)</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kappa-no-prevalence" style="text-decoration:None;">Kappa No Prevalence</a></td>
<td style="border:1px solid black;padding:4px;">-0.66667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kappa-standard-error" style="text-decoration:None;">Kappa Standard Error</a></td>
<td style="border:1px solid black;padding:4px;">0.03804</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Kappa-unbiased" style="text-decoration:None;">Kappa Unbiased</a></td>
<td style="border:1px solid black;padding:4px;">-0.04201</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Goodman-&-Kruskal's-lambda-A" style="text-decoration:None;">Lambda A</a></td>
<td style="border:1px solid black;padding:4px;">0.08333</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Goodman-&-Kruskal's-lambda-B" style="text-decoration:None;">Lambda B</a></td>
<td style="border:1px solid black;padding:4px;">0.06034</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Mutual-information" style="text-decoration:None;">Mutual Information</a></td>
<td style="border:1px solid black;padding:4px;">0.09632</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#NIR-(No-information-rate)" style="text-decoration:None;">NIR</a></td>
<td style="border:1px solid black;padding:4px;">0.2</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#Overall_ACC" style="text-decoration:None;">Overall ACC</a></td>
<td style="border:1px solid black;padding:4px;">0.16667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Overall_CEN" style="text-decoration:None;">Overall CEN</a></td>
<td style="border:1px solid black;padding:4px;">0.88621</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Overall_J" style="text-decoration:None;">Overall J</a></td>
<td style="border:1px solid black;padding:4px;">(0.46558,0.09312)</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#Overall_MCC" style="text-decoration:None;">Overall MCC</a></td>
<td style="border:1px solid black;padding:4px;">-0.04169</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Overall_MCEN" style="text-decoration:None;">Overall MCEN</a></td>
<td style="border:1px solid black;padding:4px;">0.93012</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Overall_RACC" style="text-decoration:None;">Overall RACC</a></td>
<td style="border:1px solid black;padding:4px;">0.2</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Overall_RACCU" style="text-decoration:None;">Overall RACCU</a></td>
<td style="border:1px solid black;padding:4px;">0.20027</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#P-Value" style="text-decoration:None;">P-Value</a></td>
<td style="border:1px solid black;padding:4px;">0.8706</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#PPV_Macro" style="text-decoration:None;">PPV Macro</a></td>
<td style="border:1px solid black;padding:4px;">0.16768</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#PPV_Micro" style="text-decoration:None;">PPV Micro</a></td>
<td style="border:1px solid black;padding:4px;">0.16667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Pearson's-C" style="text-decoration:None;">Pearson C</a></td>
<td style="border:1px solid black;padding:4px;">0.32066</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Phi-squared" style="text-decoration:None;">Phi-Squared</a></td>
<td style="border:1px solid black;padding:4px;">0.1146</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#RCI-(Relative-classifier-information)" style="text-decoration:None;">RCI</a></td>
<td style="border:1px solid black;padding:4px;">0.04148</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#RR-(Global-performance-index)" style="text-decoration:None;">RR</a></td>
<td style="border:1px solid black;padding:4px;">30.0</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Reference-entropy" style="text-decoration:None;">Reference Entropy</a></td>
<td style="border:1px solid black;padding:4px;">2.32193</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Response-entropy" style="text-decoration:None;">Response Entropy</a></td>
<td style="border:1px solid black;padding:4px;">2.31816</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#SOA1-(Landis-&-Koch's-benchmark)" style="text-decoration:None;">SOA1(Landis & Koch)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Red;">Poor</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#SOA2-(Fleiss'-benchmark)" style="text-decoration:None;">SOA2(Fleiss)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Red;">Poor</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#SOA3-(Altman's-benchmark)" style="text-decoration:None;">SOA3(Altman)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Red;">Poor</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#SOA4-(Cicchetti's-benchmark)" style="text-decoration:None;">SOA4(Cicchetti)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Red;">Poor</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#SOA5-(Cramer's-benchmark)" style="text-decoration:None;">SOA5(Cramer)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Orange;">Weak</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#SOA6-(Matthews's-benchmark)" style="text-decoration:None;">SOA6(Matthews)</a></td>
<td style="border:1px solid black;padding:4px;background-color:Red;">Negligible</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Scott's-Pi" style="text-decoration:None;">Scott PI</a></td>
<td style="border:1px solid black;padding:4px;">-0.04201</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Standard-error" style="text-decoration:None;">Standard Error</a></td>
<td style="border:1px solid black;padding:4px;">0.03043</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#TPR_Macro" style="text-decoration:None;">TPR Macro</a></td>
<td style="border:1px solid black;padding:4px;">0.16667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#TPR_Micro" style="text-decoration:None;">TPR Micro</a></td>
<td style="border:1px solid black;padding:4px;">0.16667</td>
</tr>
<tr align="center">
<td style="border:1px solid black;padding:4px;text-align:left;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#Zero-one-loss" style="text-decoration:None;">Zero-one Loss</a></td>
<td style="border:1px solid black;padding:4px;">125</td>
</tr>
</table>
<h2>Class Statistics : </h2>
<table style="border:1px solid black;border-collapse: collapse;">
<tr align="center">
<td>Class</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">dog</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">frog</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">giraffe</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">horse</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">tiger</td>
<td>Description</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#ACC-(Accuracy)" style="text-decoration:None;">ACC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.65333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.66667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.68</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.72</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.61333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Accuracy</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AGF-(Adjusted-F-score)" style="text-decoration:None;">AGF</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.39172</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.32663</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.4</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.47145</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.15899</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Adjusted F-score</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AGM-(Adjusted-G-mean)" style="text-decoration:None;">AGM</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.55828</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.537</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.57778</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.63226</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.42536</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Adjusted geometric mean</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AM-(Automatic/Manual)" style="text-decoration:None;">AM</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">4</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Difference between automatic and manual classification</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AUC-(Area-under-the-ROC-curve)" style="text-decoration:None;">AUC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.48333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.46667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.5</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.55</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.39583</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Area under the ROC curve</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AUCI-(AUC-value-interpretation)" style="text-decoration:None;">AUCI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">AUC value interpretation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#AUPR-(Area-under-the-PR-curve)" style="text-decoration:None;">AUPR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.18824</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1381</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.27619</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Area under the PR curve</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#BCD-(Bray-Curtis-dissimilarity)" style="text-decoration:None;">BCD</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.01333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.00667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.00667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Bray-Curtis dissimilarity</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#BM-(Bookmaker-informedness)" style="text-decoration:None;">BM</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.06667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.20833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Informedness or bookmaker informedness</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#CEN-(Confusion-entropy)" style="text-decoration:None;">CEN</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.87098</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.91505</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.86614</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.80528</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.97287</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Confusion entropy</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#DOR-(Diagnostic-odds-ratio)" style="text-decoration:None;">DOR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.82143</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.61538</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.81818</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1082</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Diagnostic odds ratio</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#DP-(Discriminant-power)" style="text-decoration:None;">DP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.0471</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.11625</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.14315</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.53245</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Discriminant power</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#DPI-(Discriminant-power-interpretation)" style="text-decoration:None;">DPI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Discriminant power interpretation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#ERR-(Error-rate)" style="text-decoration:None;">ERR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.34667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.33333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.32</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.28</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.38667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Error rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FBeta-Score" style="text-decoration:None;">F0.5</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.18072</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.14085</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.28169</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">F0.5 score</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FBeta-Score" style="text-decoration:None;">F1</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1875</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13793</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.27586</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">F1 score - harmonic mean of precision and sensitivity</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FBeta-Score" style="text-decoration:None;">F2</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.19481</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13514</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.27027</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">F2 score</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FDR-(False-discovery-rate)" style="text-decoration:None;">FDR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.82353</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.85714</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.71429</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.96667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">False discovery rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FN-(False-negative)" style="text-decoration:None;">FN</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">24</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">26</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">24</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">22</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">29</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">False negative/miss/type 2 error</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FNR-(False-negative-rate)" style="text-decoration:None;">FNR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.86667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.73333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.96667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Miss rate or false negative rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FOR-(False-omission-rate)" style="text-decoration:None;">FOR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2069</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.21311</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.18033</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.24167</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">False omission rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FP-(False-positive)" style="text-decoration:None;">FP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">28</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">24</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">24</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">20</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">29</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">False positive/type 1 error/false alarm</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#FPR-(False-positive-rate)" style="text-decoration:None;">FPR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.23333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.16667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.24167</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Fall-out or false positive rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#G-(G-measure)" style="text-decoration:None;">G</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.18787</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13801</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.27603</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">G-measure geometric mean of precision and sensitivity</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#GI-(Gini-index)" style="text-decoration:None;">GI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.06667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.20833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Gini index</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#GM-(G-mean)" style="text-decoration:None;">GM</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.39158</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.3266</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.4</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.4714</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.15899</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">G-mean geometric mean of specificity and sensitivity</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#IBA-(Index-of-balanced-accuracy)" style="text-decoration:None;">IBA</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.06644</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03556</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.064</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0963</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.00695</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Index of balanced accuracy</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#ICSI-(Individual-classification-success-index)" style="text-decoration:None;">ICSI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.62353</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.72381</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.6</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.44762</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.93333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Individual classification success index</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#IS-(Information-score)" style="text-decoration:None;">IS</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.18057</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.48543</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.51457</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-2.58496</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Information score</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#J-(Jaccard-index)" style="text-decoration:None;">J</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.10345</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.07407</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.11111</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.16</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.01695</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Jaccard index</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#LS-(Lift-score)" style="text-decoration:None;">LS</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.88235</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.71429</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.42857</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.16667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Lift score</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#MCC-(Matthews-correlation-coefficient)" style="text-decoration:None;">MCC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.03185</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.06844</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.10266</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.20833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Matthews correlation coefficient</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:aqua;"><a href="http://www.pycm.ir/doc/index.html#MCCI-(Matthews-correlation-coefficient-interpretation)" style="text-decoration:None;">MCCI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Matthews correlation coefficient interpretation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#MCEN-(Modified-confusion-entropy)" style="text-decoration:None;">MCEN</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.91864</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.95101</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.91734</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.87417</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.98141</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Modified confusion entropy</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#MK-(Markedness)" style="text-decoration:None;">MK</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.03043</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.07026</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.10539</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.20833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Markedness</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#N-(Condition-negative)" style="text-decoration:None;">N</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Condition negative</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#NLR-(Negative-likelihood-ratio)" style="text-decoration:None;">NLR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.04348</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.08333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.88</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.27473</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Negative likelihood ratio</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#NLRI-(Negative-likelihood-ratio-interpretation)" style="text-decoration:None;">NLRI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Negative likelihood ratio interpretation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#NPV-(Negative-predictive-value)" style="text-decoration:None;">NPV</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.7931</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.78689</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.81967</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.75833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Negative predictive value</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#OC-(Overlap-coefficient)" style="text-decoration:None;">OC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.14286</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.28571</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Overlap coefficient</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#OOC-(Otsuka-Ochiai-coefficient)" style="text-decoration:None;">OOC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.18787</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13801</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.27603</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Otsuka-Ochiai coefficient</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#OP-(Optimized-precision)" style="text-decoration:None;">OP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.06713</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.04762</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.08</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.20485</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.30246</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Optimized precision</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#P-(Condition-positive)" style="text-decoration:None;">P</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Condition positive or support</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#PLR-(Positive-likelihood-ratio)" style="text-decoration:None;">PLR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.85714</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.66667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1.6</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13793</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Positive likelihood ratio</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#PLRI-(Positive-likelihood-ratio-interpretation)" style="text-decoration:None;">PLRI</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Orange;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Orange;">Poor</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:Red;">Negligible</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Positive likelihood ratio interpretation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#POP-(Population)" style="text-decoration:None;">POP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">150</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">150</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">150</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">150</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">150</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Population</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#PPV-(Positive-predictive-value)" style="text-decoration:None;">PPV</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.17647</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.14286</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.28571</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Precision or positive predictive value</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#PRE-(Prevalence)" style="text-decoration:None;">PRE</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Prevalence</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Q-(Yule's-Q)" style="text-decoration:None;">Q</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.09804</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.2381</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.29032</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.80472</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Yule Q - coefficient of colligation</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#RACC-(Random-accuracy)" style="text-decoration:None;">RACC</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04533</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03733</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03733</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Random accuracy</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#RACCU-(Random-accuracy-unbiased)" style="text-decoration:None;">RACCU</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04551</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03738</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03738</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.04</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Random accuracy unbiased</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TN-(True-negative)" style="text-decoration:None;">TN</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">92</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">96</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">96</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">100</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">91</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">True negative/correct rejection</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TNR-(True-negative-rate)" style="text-decoration:None;">TNR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.76667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.83333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.75833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Specificity or true negative rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TON-(Test-outcome-negative)" style="text-decoration:None;">TON</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">116</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">122</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">122</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">120</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Test outcome negative</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TOP-(Test-outcome-positive)" style="text-decoration:None;">TOP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">34</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">28</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">28</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">30</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Test outcome positive</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TP-(True-positive)" style="text-decoration:None;">TP</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">6</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">4</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">6</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">8</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">1</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">True positive/hit</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#TPR-(True-positive-rate)" style="text-decoration:None;">TPR</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.13333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.2</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.26667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Sensitivity, recall, hit rate, or true positive rate</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#Y-(Youden-index)" style="text-decoration:None;">Y</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.03333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.06667</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.0</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.1</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">-0.20833</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Youden index</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#dInd-(Distance-index)" style="text-decoration:None;">dInd</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.83333</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.88944</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.82462</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.75203</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.99642</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Distance index</td>
</tr>
<tr align="center" style="border:1px solid black;border-collapse: collapse;">
<td style="border:1px solid black;padding:4px;border-collapse: collapse;background-color:transparent;"><a href="http://www.pycm.ir/doc/index.html#sInd-(Similarity-index)" style="text-decoration:None;">sInd</a></td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.41074</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.37107</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.4169</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.46823</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;">0.29543</td>
<td style="border:1px solid black;padding:4px;border-collapse: collapse;text-align:left;">Similarity index</td>
</tr>
</table>
</body>
<p style="text-align:center;border-top:1px solid black;">Generated By <a href="http://www.pycm.ir" style="text-decoration:none;color:red;">PyCM</a> Version 2.5</p>
</html>






```
References to the library: https://www.pycm.ir/doc/index.html#Cite

  @article{Haghighi2018,
  doi = {10.21105/joss.00729},
  url = {https://doi.org/10.21105/joss.00729},
  year  = {2018},
  month = {may},
  publisher = {The Open Journal},
  volume = {3},
  number = {25},
  pages = {729},
  author = {Sepand Haghighi and Masoomeh Jasemi and Shaahin Hessabi and Alireza Zolanvari},
  title = {{PyCM}: Multiclass confusion matrix library in Python},
  journal = {Journal of Open Source Software}
  }
```
