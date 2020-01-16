import sys
import os
import argparse
root_path = "/home/developer/Desktop/Classification_Pipeline_Classes/"
pkg_path = root_path + "utils/"
sys.path.insert(1, pkg_path)
from training import TrainingUtils

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_dirs(params):
    """
    This function will be used to create the neccessary data directories,
    which will be further used by the pipeline to save the trained models,
    evaluation results, hyperparameters, training logs etc.
    
        Arguments:
            
            -params  : The list of input params that the user has entered 
                       through the terminal. If nothing is entered, the
                       default values present in the argument parser will
                       be passed as input params.                    
    """
    
    SIM_NUM = params['sim']
    
    #This dictionary, path_dict will contain all the locations of evaluation results and model training.
    os.mkdir(root_path + "simulations/") if not os.path.isdir(root_path + "simulations/") else None
    os.mkdir(root_path + "data/") if not os.path.isdir(root_path + "data/") else None    

    path_dict = dict()
    path_dict['sim_path'] = root_path + "simulations/" + "SIM_{:02d}/".format(SIM_NUM)
    
    if(os.path.isdir(path_dict['sim_path']) == False):
        os.mkdir(path_dict['sim_path'])
    else:
        print("\nPlease enter a different simulation number as the folder SIM_{:02d} already exists! \n".format(SIM_NUM))
        NEW_SIM_NUM = int(input("Enter a new simulation number! \n"))
        input_params['sim'] = NEW_SIM_NUM
        path_dict['sim_path'] = root_path + "simulations/" + "SIM_{:02d}/".format(NEW_SIM_NUM)
        os.mkdir(path_dict['sim_path'])
            
    path_dict['model_path'] = path_dict['sim_path'] + "training_results/"
    path_dict['eval_path'] = path_dict['sim_path'] + "evaluation_results/"
    path_dict['weights_path'] = root_path + "weights/"
    path_dict['source'] = root_path + "data/"
    path_dict['df_path'] = root_path + "data_df/"
    
    os.mkdir(path_dict['model_path']) if not os.path.isdir(path_dict['model_path']) else None
    os.mkdir(path_dict['model_path'] + "stage1/") if not os.path.isdir(path_dict['model_path'] + "stage1/") else None
    os.mkdir(path_dict['model_path'] + "stage2/") if not os.path.isdir(path_dict['model_path'] + "stage2/") else None
         
    os.mkdir(path_dict['eval_path']) if not os.path.isdir(path_dict['eval_path']) else None
    os.mkdir(path_dict['eval_path'] + "stage1/") if not os.path.isdir(path_dict['eval_path'] + "stage1/") else None
    os.mkdir(path_dict['eval_path'] + "stage2/") if not os.path.isdir(path_dict['eval_path'] + "stage2/") else None
    
    return path_dict, input_params 

if __name__ == '__main__':
    """
    List of all the arguments that the user will need to pass to the pipeline in order 
    to get started with the model training and evaluation phases. If no value is given
    by the user, the default arguments will be sent to the pipeline as input params. 
    Please note, that it is mandatory to provide the SIMULATION NUMBER as a parameter
    every time you run this code -> [The argument "--sim"]
    """
    
    parser = argparse.ArgumentParser(description='this script will train 2 machine learning models using transfer learning')
    parser.add_argument('--model_name', type=str, default='inceptionv3', help='choose the type of model you want to train with')
    parser.add_argument('--dense_neurons', type=int, default=96, help='eneter the number of neurons you want for the pre-final layer')
    parser.add_argument('--batch_size', type=int, default=32, help="enter the number of batches for which the model should be trained on")
    parser.add_argument('--stage1_lr', type=float, default=0.0005, help="enter the learning rate for stage 1 training")
    parser.add_argument('--stage2_lr', type=float, default=0.00001, help="enter the learning rate for stage 2 training")
    parser.add_argument('--monitor',type=str, default='val_accuracy', help="enter the metric you want to monitor")
    parser.add_argument('--metric',type=str, default='accuracy', help="enter the metric you want the model to optimize")
    parser.add_argument('--epochs1',type=int, default=10, help="enter the number of epochs you want the model to train for in stage 1")
    parser.add_argument('--epochs2',type=int, default=10, help="enter the number of epochs you want the model to train for in stage 2")
    parser.add_argument('--sim',type=int, default=None, help="enter the simulation number")
    parser.add_argument('--finetune',type=str, default='yes', help="state 'yes' or 'no' to say whether or not you want to fine tune the convolution block")
    args = parser.parse_args()

    #All the inputs to the pipeline will be saved in this dictionary input_params, which will be passed to the 
    #subsequent function calls for training and evaluation of the Machine Learning pipeline.
    input_params = dict()
    input_params['sim'] = args.sim
    input_params['model_name'] = args.model_name
    input_params['dense_neurons'] = args.dense_neurons
    input_params['batch_size'] = args.batch_size
    input_params['stage1_lr'] = args.stage1_lr
    input_params['stage2_lr'] = args.stage2_lr
    input_params['monitor'] = args.monitor
    input_params['metric'] = args.metric
    input_params['epochs1'] = args.epochs1
    input_params['epochs2'] = args.epochs2
    input_params['finetune'] = args.finetune
    
    #Create the directories
    path_dict, input_params = create_dirs(input_params)
    
    #Create an instance variable of TrainingUtils class
    train_utils_obj = TrainingUtils(input_params, path_dict)
    
    #Save the hyperparameters
    train_utils_obj.save_params()
    
    #Start model training and evaluation
    train_utils_obj.train()
