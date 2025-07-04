import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import json
from utils import get_time_dif
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
# from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, args, class_weight):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=config.learning_rate)
    # 单独设置warmup调度器
    total_steps = len(train_iter) * config.num_epochs
    warmup_steps = int(total_steps * 0.05)  # 5%的warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    total_batch = 0  # 记录进行到多少batch
    flag = False
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                p, r, f1, dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    print('model have saved')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                msg = 'Iter: {:>6},  Val P: {:>5.4},  Val R: {:>6.4%},  Val F1: {:>5.4},  Val Acc: {:>6.4%}, Train Acc: {:>6.4%} Time: {} {}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, train_acc, time_dif, improve))
                print(msg.format(total_batch, p, r, f1, dev_acc, train_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter, args)


def test(config, model, test_iter, args):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    evaluate(config, model, test_iter, test=True)
    # msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    # msg = 'Test P: {:>6.4}, Test R: {:>6.4}, Test F: {:>6.4},  Test Acc: {:>6.4%}'
    # print(msg.format(test_loss, test_acc))
    """
    print(msg.format(p, r, f1, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    if args.save_path is not None:
        print("saving predictions to {}.".format(args.save_path))
        # np.save(args.save_path, predict_all)
        np.savez(args.save_path, test_confusion=test_confusion, test_prediction=predict_all)
    """


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    p = metrics.precision_score(labels_all, predict_all, average='macro')
    r = metrics.recall_score(labels_all, predict_all, average='macro')
    f1 = metrics.f1_score(labels_all, predict_all, average='macro')
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        """
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # return acc, loss_total / len(data_iter), report, confusion, predict_all
        return p, r, f1, acc, loss_total / len(data_iter), report, confusion, predict_all
        """
        with open(r'E:\LJ\task4\data\IMCS-DAC_test.json', 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        # 2. 确保预测结果数量与数据匹配
        assert len(predict_all) == sum(len(dialog) for dialog in original_data.values()), \
            "预测结果数量与测试数据不匹配"

        # 3. 将预测结果写回dialogue_act字段
        idx = 0
        for dialog_id in original_data:
            for sentence in original_data[dialog_id]:
                # 获取对应的类别名称
                pred_label = config.class_list[predict_all[idx]]
                sentence['dialogue_act'] = pred_label
                idx += 1

        # 4. 保存更新后的JSON文件
        output_path = r'E:\LJ\task4\BERT-DAC\data\IMCS-DAC_test.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)

        print(f"预测结果已写入: {output_path}")
    # return acc, loss_total / len(data_iter)
    return p, r, f1, acc, loss_total / len(data_iter)
