import pdb 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


#질문 생성 모델 load
peft_model_id = "./outputs_qg/checkpoint-1000"  #Question Generation 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
qg_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
qg_model = PeftModel.from_pretrained(qg_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

qg_model.eval()

# 답변 생성 모델 load 
peft_model_id = "./outputs_ag/checkpoint-1000"  #Answer Generation 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
ag_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
ag_model = PeftModel.from_pretrained(ag_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

ag_model.eval()

#질문 생성 모델 output 함수(input: context, output: question)
def qg_output(context):
    q = f"### 문맥: {context}\n\n위의 문맥으로부터 정답을 찾을 수 있는 질문 하나만 만들어줘\n\n### 질문:"
    # print(q)
    gened = qg_model.generate(
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
    question_start_index = result.find('질문:')
    question_end_index = question_start_index + result[question_start_index:].find('?')
    return result[question_start_index+3: question_end_index+1]

#답변 생성 모델 output 함수(input: context, question  output: answer)
def ag_output(context, question):
    q = f"### 문맥: {context}\n\n위의 문맥으로부터 다음 질문의 정답을 찾아라 ### 질문: {question}\n\n### 정답:"
    # print(q)
    gened = ag_model.generate(
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

# 위의 두 함수를 조합한 최종 모델 output 함수 (input: context  output: qusetion, answer)
def test(context):
    question = qg_output(context) 
    answer = ag_output(context, question)
    print( f"### 문맥: {context}\n\n### 질문: {question}\n\n### 정답: {answer}")

test('운영체제는 파일 시스템 관리, 프로세스 스케줄링, 메모리 관리, 동기화 및 상호배제 등의 다양한 기능을 수행한다.')
test('밴쿠버는 캐나다 서쪽에 있는 작은 항구 도시야. 배를 타고 밴쿠버에 가면 산들이 병풍처럼 도시를 감싸고 있는 게 보여.파란 바다와 산, 맑은 하늘이 만나 하늘이 만나 빚어내는 풍경은 눈부시게 아름답지!밴쿠버에는 걸어 서도 푸른 숲과 깨끗한 바다에 갈 수 있어. ')
