import time
import torch
import numpy as np
from train_eval import train, init_network,test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from models.bert import Config, Model

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='Bert', help='choose a model: Bert, ERNIE')
parser.add_argument('--save_path', type=str, required=False, help='the save path of predictions on test set')
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    dataset = ''  # 数据集

    model_name = args.model  # bert
    print(model_name)
    #x = import_module('models.' + model_name)
    config = Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    class_count = {'15': 24662, '1': 9945, '0': 6909, '9': 5442, '7': 4154, '13': 3735, '5': 3375, '6': 2781, '8': 2692,
                   '11': 2638, '4': 2158, '14': 1702, '3': 1198, '10': 842, '2': 714, '12': 656}

    class_counts_tensor = torch.tensor([class_count[str(i)] for i in range(len(class_count))], dtype=torch.float32)

    # 计算类别权重
    total_samples = class_counts_tensor.sum()
    class_weight = total_samples / (len(class_count) * class_counts_tensor)
    # print(class_weight)  tensor([0.6658, 0.4626, 6.4428, 3.8399, 2.1317, 1.3630, 1.6541, 1.1074, 1.7088,
    #         0.8453, 5.4634, 1.7438, 7.0125, 1.2316, 2.7028, 0.1865])
    # class_weight = (class_weight - class_weight.min()) / (class_weight.max() - class_weight.min())
    # train
    class_weight=class_weight.to(config.device)
    model = Model(config).to(config.device)
    # 'dac_predictions.npy'
    train(config, model, train_iter, dev_iter, test_iter, args, class_weight)
    # test(config, model, test_iter, args)
