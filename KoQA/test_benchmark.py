#pretrained 모델과 fine tuned 모델의 성능을 정량적으로 비교하기 위한 코드
#네이버 영화 리뷰 감정 평가 task에 대한 성능을 비교함.  

'''
bash  shell에서 다음 명령어들로 데이터 다운받기 
mkdir -p data_in/KOR/naver_movie
wget https://raw.githubusercontent.com/NLP-kr/tensorflow-ml-nlp-tf2/master/7.PRETRAIN_METHOD/data_in/KOR/naver_movie/ratings_train.txt \
              -O data_in/KOR/naver_movie/ratings_train.txt
wget https://raw.githubusercontent.com/NLP-kr/tensorflow-ml-nlp-tf2/master/7.PRETRAIN_METHOD/data_in/KOR/naver_movie/ratings_test.txt \
              -O data_in/KOR/naver_movie/ratings_test.txt

'''

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

import torch
import pandas as pd
import numpy as np
import re

import random
from random import sample

from tqdm import tqdm

SEED_NUM = 1234
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)

# fine tuned 모델 load 
peft_model_id = "./outputs/checkpoint-500"  #finetuned 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
cls_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
cls_model = PeftModel.from_pretrained(cls_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


cls_model.config.max_length = 2048
cls_model.config.pad_token_id = 0



# 데이터 전처리 준비
DATA_IN_PATH = './data_in/KOR'
DATA_OUT_PATH = './data_out/KOR'

DATA_TRAIN_PATH = os.path.join(DATA_IN_PATH, 'naver_movie', 'ratings_train.txt')
DATA_TEST_PATH = os.path.join(DATA_IN_PATH, 'naver_movie', 'ratings_test.txt')

train_data = pd.read_csv(DATA_TRAIN_PATH, header = 0, delimiter = '\t', quoting = 3)
train_data = train_data.dropna()

print('데이터 positive 라벨: ', '긍정')
print('데이터 negative 라벨: ', '부정')
print('학습 예시 케이스 구조: ', '문장: 오늘 기분이 좋아\n감정: 긍정\n')
print('gpt 최대 토큰 길이: ', cls_model.config.max_position_embeddings)

sent_lens = [len(tokenizer(s).input_ids) for s in tqdm(train_data['document'])]

print('Few shot 케이스 토큰 평균 길이: ', np.mean(sent_lens))
print('Few shot 케이스 토큰 최대 길이: ', np.max(sent_lens))
print('Few shot 케이스 토큰 길이 표준편차: ',np.std(sent_lens))
print('Few shot 케이스 토큰 길이 80 퍼센타일: ',np.percentile(sent_lens, 80))

train_fewshot_data = []

for train_sent, train_label in tqdm(train_data[['document', 'label']].values):
    tokens = tokenizer(train_sent).input_ids

    if len(tokens) <= 25:
        train_fewshot_data.append((train_sent, train_label))


test_data = pd.read_csv(DATA_TEST_PATH, header=0, delimiter='\t', quoting=3)
test_data = test_data.dropna()
test_data.head()

# Full Dataset
# sample_size = len(test_data)

# Sampled Dataset
sample_size = 500

train_fewshot_samples = []

for _ in range(sample_size):
    fewshot_examples = sample(train_fewshot_data, 10)
    train_fewshot_samples.append(fewshot_examples)

if sample_size < len(test_data['id']):
    test_data = test_data.sample(sample_size, random_state=SEED_NUM)





#finetuned 모델 프롬프트 1 방식 테스트 

def build_prompt_text(sent):
    return "문장: " + sent + '\n감정:'

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sent)
    return sent_clean

real_labels = []
pred_tokens = []

total_len = len(test_data[['document','label']].values)

for i, (test_sent, test_label) in tqdm(enumerate(test_data[['document','label']].values), total=total_len):
    prompt_text = ''

    for ex in train_fewshot_samples[i]:
        example_text, example_label = ex
        cleaned_example_text = clean_text(example_text)
        appended_prompt_example_text = build_prompt_text(cleaned_example_text)
        appended_prompt_example_text += ' 긍정' if example_label == 1 else ' 부정' + '\n'
        prompt_text += appended_prompt_example_text

    cleaned_sent = clean_text(test_sent)
    appended_prompt_sent = build_prompt_text(cleaned_sent)

    prompt_text += appended_prompt_sent

    tokens = tokenizer(prompt_text, return_tensors="pt")
    token_ids, attn_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
    gen_tokens = cls_model.generate(input_ids=token_ids, attention_mask=attn_mask,
                                    max_new_tokens=1, pad_token_id=0)
    pred = tokenizer.batch_decode(gen_tokens[:, -1])[0].strip()

    pred_tokens.append(pred)
    real_labels.append('긍정' if test_label == 1 else '부정')

accuracy_match = [p == t for p, t in zip(pred_tokens, real_labels)]
accuracy = len([m for m in accuracy_match if m]) / len(real_labels)

print('finetuned 모델 프롬프트 1 방식 테스트: ', accuracy)


#finetuned 모델 프롬프트2  방식 테스트 
def build_prompt_text(sent):
    return '다음 문장은 긍정일까요 부정일까요?\n' + sent + '\n정답:'

real_labels = []
pred_tokens = []


real_labels = []
pred_tokens = []

total_len = len(test_data[['document','label']].values)

for i, (test_sent, test_label) in tqdm(enumerate(test_data[['document','label']].values), total=total_len):
    prompt_text = ''

    for ex in train_fewshot_samples[i]:
        example_text, example_label = ex
        cleaned_example_text = clean_text(example_text)
        appended_prompt_example_text = build_prompt_text(cleaned_example_text)
        appended_prompt_example_text += ' 긍정' if example_label == 1 else ' 부정' + '\n'
        prompt_text += appended_prompt_example_text

    cleaned_sent = clean_text(test_sent)
    appended_prompt_sent = build_prompt_text(cleaned_sent)

    prompt_text += appended_prompt_sent

    tokens = tokenizer(prompt_text, return_tensors="pt")
    token_ids, attn_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
    gen_tokens = cls_model.generate(input_ids=token_ids, attention_mask=attn_mask,
                                    max_new_tokens=1, pad_token_id=0)
    pred = tokenizer.batch_decode(gen_tokens[:, -1])[0].strip()

    pred_tokens.append(pred)
    real_labels.append('긍정' if test_label == 1 else '부정')

accuracy_match = [p == t for p, t in zip(pred_tokens, real_labels)]
accuracy = len([m for m in accuracy_match if m]) / len(real_labels)

print('finetuned 모델 프롬프트2  방식 테스트: ', accuracy)


# pretrained 모델 프롬프트 1 방식 테스트 
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")

#hugging face에서 pretrained 모델 load
cls_model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-5.8b",
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True).cuda()

def build_prompt_text(sent):
    return "문장: " + sent + '\n감정:'

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sent)
    return sent_clean

real_labels = []
pred_tokens = []

total_len = len(test_data[['document','label']].values)

for i, (test_sent, test_label) in tqdm(enumerate(test_data[['document','label']].values), total=total_len):
    prompt_text = ''

    for ex in train_fewshot_samples[i]:
        example_text, example_label = ex
        cleaned_example_text = clean_text(example_text)
        appended_prompt_example_text = build_prompt_text(cleaned_example_text)
        appended_prompt_example_text += ' 긍정' if example_label == 1 else ' 부정' + '\n'
        prompt_text += appended_prompt_example_text

    cleaned_sent = clean_text(test_sent)
    appended_prompt_sent = build_prompt_text(cleaned_sent)

    prompt_text += appended_prompt_sent

    tokens = tokenizer(prompt_text, return_tensors="pt")
    token_ids, attn_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
    gen_tokens = cls_model.generate(input_ids=token_ids, attention_mask=attn_mask,
                                    max_new_tokens=1, pad_token_id=0)
    pred = tokenizer.batch_decode(gen_tokens[:, -1])[0].strip()

    pred_tokens.append(pred)
    real_labels.append('긍정' if test_label == 1 else '부정')

accuracy_match = [p == t for p, t in zip(pred_tokens, real_labels)]
accuracy = len([m for m in accuracy_match if m]) / len(real_labels)

print('pretrained 모델 프롬프트 1 방식 테스트: ', accuracy)



# pretrained 모델 프롬프트 2 방식 테스트 
def build_prompt_text(sent):
    return '다음 문장은 긍정일까요 부정일까요?\n' + sent + '\n정답:'

real_labels = []
pred_tokens = []


real_labels = []
pred_tokens = []

total_len = len(test_data[['document','label']].values)

for i, (test_sent, test_label) in tqdm(enumerate(test_data[['document','label']].values), total=total_len):
    prompt_text = ''

    for ex in train_fewshot_samples[i]:
        example_text, example_label = ex
        cleaned_example_text = clean_text(example_text)
        appended_prompt_example_text = build_prompt_text(cleaned_example_text)
        appended_prompt_example_text += ' 긍정' if example_label == 1 else ' 부정' + '\n'
        prompt_text += appended_prompt_example_text

    cleaned_sent = clean_text(test_sent)
    appended_prompt_sent = build_prompt_text(cleaned_sent)

    prompt_text += appended_prompt_sent

    tokens = tokenizer(prompt_text, return_tensors="pt")
    token_ids, attn_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
    gen_tokens = cls_model.generate(input_ids=token_ids, attention_mask=attn_mask,
                                    max_new_tokens=1, pad_token_id=0)
    pred = tokenizer.batch_decode(gen_tokens[:, -1])[0].strip()

    pred_tokens.append(pred)
    real_labels.append('긍정' if test_label == 1 else '부정')

accuracy_match = [p == t for p, t in zip(pred_tokens, real_labels)]
accuracy = len([m for m in accuracy_match if m]) / len(real_labels)

print('pretrained 모델 프롬프트 2 방식 테스트: ', accuracy)