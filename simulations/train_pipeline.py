import sys
root_path = "/home/developer/Desktop/Saugata/Classification-Pipeline/"
pkg_path = root_path + "utils/"
sys.path.insert(1, pkg_path)
import training
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



def create_dir(params):
    
    SIM_NUM = params['sim']

    path_dict = dict()
    
    path_dict['sim_path'] = root_path + "simulations/" + "SIM_{:02d}/".format(SIM_NUM)
    path_dict['model_path'] = path_dict['sim_path'] + "models/"
    path_dict['eval_path'] = path_dict['sim_path'] + "evaluation_results/"
    
    path_dict['weights_path'] = root_path + "weights/"
    path_dict['source'] = root_path + "data/"
    path_dict['df_path'] = root_path + "data_df/"
    
    os.mkdir(path_dict['sim_path']) if not os.path.isdir(path_dict['sim_path']) else None
    
    os.mkdir(path_dict['model_path']) if not os.path.isdir(path_dict['model_path']) else None
    os.mkdir(path_dict['model_path'] + "stage1/") if not os.path.isdir(path_dict['model_path'] + "stage1/") else None
    os.mkdir(path_dict['model_path'] + "stage2/") if not os.path.isdir(path_dict['model_path'] + "stage2/") else None
         
    os.mkdir(path_dict['eval_path']) if not os.path.isdir(path_dict['eval_path']) else None
    os.mkdir(path_dict['eval_path'] + "stage1/") if not os.path.isdir(path_dict['eval_path'] + "stage1/") else None
    os.mkdir(path_dict['eval_path'] + "stage2/") if not os.path.isdir(path_dict['eval_path'] + "stage2/") else None
    
    return path_dict 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this script will train 2 machine learning models using transfer learning')
    parser.add_argument('--model_name', type=str, default='vgg16', help='choose the type of model you want to train with')
    parser.add_argument('--dense_neurons', type=int, default=64, help='eneter the number of neurons you want for the pre-final layer')
    parser.add_argument('--batch_size', type=int, default=32, help="enter the number of batches for which the model should be trained on")
    parser.add_argument('--stage1_lr', type=float, default=0.0005, help="enter the learning rate for stage 1 training")
    parser.add_argument('--stage2_lr', type=float, default=0.00001, help="enter the learning rate for stage 2 training")
    parser.add_argument('--monitor',type=str, default='val_accuracy', help="enter the metric you want to monitor")
    parser.add_argument('--metric',type=str, default='accuracy', help="enter the metric you want the model to optimize")
    parser.add_argument('--epochs1',type=int, default=1, help="enter the number of epochs you want the model to train for in stage 1")
    parser.add_argument('--epochs2',type=int, default=1, help="enter the number of epochs you want the model to train for in stage 2")
    parser.add_argument('--sim',type=int, default=None, help="enter the simulation number")
    parser.add_argument('--finetune',type=str, default='yes', help="state 'yes' or 'no' to say whether or not you want to fine tune the convolution block")
    args = parser.parse_args()

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
    path_dict = create_dir(input_params)
    
    #Save all the hyperparameters
    training.save_params(input_params, path_dict)

    #Start training + evaluation on unseen data
    training.train(input_params, path_dict)
