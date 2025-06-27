import json
import os
import pandas as pd
from collections import defaultdict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data




train_set = load_json(r'E:\LJ\task4\data\IMCS-DAC_train.json')
dev_set = load_json(r'E:\LJ\task4\data\IMCS-DAC_dev.json')
test_set = load_json(r'E:\LJ\task4\data\fill_IMCS-DAC_test.json')

saved_path = 'data'
os.makedirs(saved_path, exist_ok=True)
tags = [
    'Request-Symptom', 'Inform-Symptom', 'Request-Etiology', 'Inform-Etiology', 'Request-Basic_Information',
    'Inform-Basic_Information', 'Request-Existing_Examination_and_Treatment',
    'Inform-Existing_Examination_and_Treatment',
    'Request-Drug_Recommendation', 'Inform-Drug_Recommendation', 'Request-Medical_Advice',
    'Inform-Medical_Advice', 'Request-Precautions', 'Inform-Precautions',
    'Diagnose', 'Other'
]
tag2id = {tag: idx for idx, tag in enumerate(tags)}
#print(tag2id)


def make_tag(path):
    with open(path, 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write(tag + '\n')


def make_data(samples, path):
    out = ''
    for pid, sample in samples.items():
        #print(sample)
        for sent in sample:
            x = sent['speaker'] + '：' + sent['sentence']
            #print(sent['dialogue_act'],sent['sentence_id'],pid)
            assert sent['dialogue_act'] in tag2id
            y = tag2id.get(sent['dialogue_act'])
            out += x + '\t' + str(y) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return out
"""
def create_samples(dialogue_df, window_size=7):
    samples = []
    for i in range(len(dialogue_df)):
        context = dialogue_df[max(0,i-window_size):i]
        samples.append({
            'text': dialogue_df.iloc[i]['text'],
            'context': ' [SEP] '.join([f"{row['text']}" for _,row in context.iterrows()]),
            'label': dialogue_df.iloc[i]['label']
        })
    return pd.DataFrame(samples)
"""


def create_samples(dialogue_data, window_size=2):
    samples = []

    # First, convert the dialogue data into a list of dictionaries with pid information
    all_sentences = []
    for pid, dialogue in dialogue_data.items():
        for sent in dialogue:
            all_sentences.append({
                'pid': pid,
                'text': sent['speaker'] + '：' + sent['sentence'],
                'label': tag2id.get(sent['dialogue_act']),
                'sentence_id': int(sent['sentence_id'])
            })

    # Group sentences by pid
    dialogues = defaultdict(list)
    for sent in all_sentences:
        dialogues[sent['pid']].append(sent)

    # Sort each dialogue by sentence_id
    for pid in dialogues:
        dialogues[pid] = sorted(dialogues[pid], key=lambda x: x['sentence_id'])

    # Create samples for each dialogue separately
    for pid, dialogue in dialogues.items():
        for i in range(len(dialogue)):
            # Get context from current dialogue only
            context_start = max(0, i - window_size)
            context = dialogue[context_start:i]

            samples.append({
                'text': dialogue[i]['text'],
                'context': ' [SEP] '.join([sent['text'] for sent in context]),
                'label': dialogue[i]['label']
            })

    return pd.DataFrame(samples)


for dataset_name, dataset in [('train', train_set), ('dev', dev_set), ('test', test_set)]:
    pro_data = create_samples(dataset)
    output_path = os.path.join(r'E:\LJ\task4\BERT-DAC\data', f'{dataset_name}_with_context1.txt')

    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in pro_data.iterrows():
            line = f"{row['context']}[SEP]{row['text']}\t{row['label']}\n"
            f.write(line)

    print(f"成功保存{len(pro_data)}个样本到{output_path}")

# 读取数据
# 假设每行的格式是 "角色：内容\t标签"
# 使用 Pandas 的 read_csv 方法读取，分隔符为制表符 (\t)
#with open(r'E:\LJ\task4\BERT-DAC\data\test.txt','r',encoding='utf-8') as data:
    #df = pd.read_csv(data, sep='\t', header=None, names=['text', 'label'])
#pro_data=create_samples((df))

#with open(r'E:\LJ\task4\BERT-DAC\data\test1.txt', 'w', encoding='utf-8') as f:
    #for _, row in pro_data.iterrows():
        # 构造每行内容
        #line = f"{row['context']}[SEP]{row['text']}\t{row['label']}\n"
        #f.write(line)

#print(f"成功保存{len(pro_data)}个样本")



#make_tag(os.path.join(saved_path, 'class.txt'))

#make_data(train_set, os.path.join(saved_path, 'train.txt'))
#make_data(dev_set, os.path.join(saved_path, 'dev.txt'))
#make_data(test_set, os.path.join(saved_path, 'test.txt'))
