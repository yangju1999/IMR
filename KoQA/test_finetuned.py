# instruction tuning을 완료한 모델 test를 위한 코드 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


#peft 방식으로 fine tuning 한 모델 불러오기  
peft_model_id = "./outputs/checkpoint-500"  #finetuned 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(    #test할때도 동일하게 4비트 양자화 사용 
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()

#모델 테스트를 위한 함수 (질문을 input으로 받아 answer를 output)
def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
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
    print(tokenizer.decode(gened[0]))

gen('건강하게 살기 위한 세 가지 방법은?')
gen('피부가 좋아지는 방법은?')
gen('파이썬과 c 언어중에 코드 실행속도 측면에서 빠른 언어는?')
gen('축구를 가장 잘하는 나라는?')