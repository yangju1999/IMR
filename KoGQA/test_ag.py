import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


#peft 방식으로 fine tuning 한 모델 불러오기  
peft_model_id = "./outputs_ag/checkpoint-1000"  #finetuned 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(  #test할때도 동일하게 4비트 양자화 사용 
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
    result = tokenizer.decode(gened[0])  #깔끔하게 정답 1개만을 출력하기 위한 전처리 코드 
    answer_start_index = result.find('정답:')
    answer_end_index = answer_start_index + result[answer_start_index:].find('.')
    print(result[answer_start_index+3: answer_end_index+1])

gen('2020∼2022년 약 3년간 지구촌을 강타한 신종 코로나바이러스 감염증(코로나19)의 미스터리 중 하나는 감염돼도 증상이 없는 ‘무증상 감염자’가 존재한다는 점이었다. 감기로 오인할 만한 증상이 나타나거나 위중증에 이르는 사람과는 확연히 다른 무증상 감염자는 방역을 더욱 어렵게 만든 원인이었다.', 
'약 3년간 지구촌을 강타한 신종 코로나바이러스 감염증(코로나19)의 미스터리 중 하나는?')