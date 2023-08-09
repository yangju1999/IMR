# 정답 생성 모델 정량적 평가를 위한 코드 (질문 생성 모델은 정량적 평가 어려움)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import pdb

#peft 방식으로 fine tuning 한 Answer Generation모델 불러오기
peft_model_id = "./outputs_ag/checkpoint-1000"  #finetuned 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()

#모델 테스트를 위한 함수 (문맥과 질문을 input으로 받아 answer를 output)
def gen(context, question):
    q = f"### 문맥: {context}\n\n위의 문맥으로부터 다음 질문의 정답을 찾아라 ### 질문: {question}\n\n### 정답:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=100,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    result = tokenizer.decode(gened[0])
    answer_start_index = result.find('정답:')
    answer_end_index = answer_start_index + result[answer_start_index:].find('.')
    return result[answer_start_index+3: answer_end_index+1]


#test를 위한 데이터 셋 load 
from datasets import load_dataset

#huggingface에서 squad_kor_v1 데이터 셋 가져오기 
test_data = load_dataset("squad_kor_v1")
test_data = test_data['validation'] 
test_data = test_data[:10] #10개 sample만 테스트 진행 

real_labels = [] #실제 정답 저장을 위한 리스트 
pred_tokens = []#모델이 생성한 정답을 저장하기 위한 리스트 

total_len = len(test_data['context'])

# test 데이터 셋에서 문맥, 질문, 답변 쌍을 반복문으로 가져오고, 
# 실제 답변은 real 리스트에 저장하고, 모델에 문맥,질문을 집어넣어서 생성된 output은 pred 리스트에 저장. 
for test_context, test_question, test_answer in tqdm(zip(test_data['context'], test_data['question'], test_data['answers']), total= total_len):

    test_answer = test_answer['text'][0]
    answer = gen(test_context, test_question)

    pred_tokens.append(answer)
    real_labels.append(test_answer)

#실제 정답이 모델이 생성한 정답에 포함이 되어있으면 True, 없으면 False로 간주 
accuracy_match = [True if t in p else False for p, t in zip(pred_tokens, real_labels)]
#전체에서 True 의 비율로 Accuracy 측정 
accuracy = len([m for m in accuracy_match if m]) / len(real_labels)

print('Answer Generation 모델 정량적 평과 결과: ', accuracy)