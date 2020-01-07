import training
import evaluation
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='this script will train 3 machine learning modelsusing transfer learning')
    parser.add_argument('--model_name', type=str, default='vgg16', help='choose the type of model you want to train with')
    parser.add_argument('--dense_neurons', type=int, default=64, help='eneter the number of neurons you want for the pre-final layer')
    parser.add_argument('--batch_size', type=int, default=32, help="enter the number of batches for which the model should be trained on")
    parser.add_argument('--stage1_lr', type=float, default=0.0005, help="enter the learning rate for stage 1 training")
    parser.add_argument('--stage2_lr', type=float, default=0.00001, help="enter the learning rate for stage 2 training")
    parser.add_argument('--monitor',type=str, default='val_accuracy', help="enter the metric you want to monitor")
    parser.add_argument('--metric',type=str, default='accuracy', help="enter the metric you want the model to optimize")
    parser.add_argument('--epochs1',type=int, default=2, help="enter the number of epochs you want the model to train for in stage 1")
    parser.add_argument('--epochs2',type=int, default=2, help="enter the number of epochs you want the model to train for in stage 2")
    parser.add_argument('--finetune',type=str, default='yes', help="state 'yes' or 'no' to say whether or not you want to fine tune the convolution block")
    args = parser.parse_args()

    input_params = dict()
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
    
    training.save_params(input_params)

    training.train(input_params)

