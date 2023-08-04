import pdb 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_id = "./outputs_qg/checkpoint-500"  #Question Generation 모델 path  
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


peft_model_id = "./outputs_qa/checkpoint-500"  #Question Answer 모델 path  
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
qa_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
qa_model = PeftModel.from_pretrained(qa_model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

qa_model.eval()


def qg_output(context):
    q = f"### 문맥: {context}\n\n### 질문:"
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
    return tokenizer.decode(gened[0])


def final_output(context, question):
    q = f"### 문맥: {context}\n\n### 질문: {question}\n\n### 답변:"
    # print(q)
    gened = qa_model.generate(
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
    answer_start_index = tokenizer.decode(gened[0]).find('답변:')
    if tokenizer.decode(gened[0])[answer_start_index:].find('###') != -1:
        answer_end_index = answer_start_index + tokenizer.decode(gened[0])[answer_start_index:].find('###')
        print(tokenizer.decode(gened[0])[:answer_end_index-1])
    else:
        print(tokenizer.decode(gened[0]))


def test(context):
    context_question = qg_output(context) #question genertion model의 output인 context, question

    #context_question 전처리 (문맥과 질문 string 분리)
    context_start_index = context_question.find('문맥:')
    question_start_index = context_question.find('질문:')
    question_end_index = context_question.find('?')

    context_str = context_question[context_start_index+4:question_start_index-4]
    question_str = context_question[question_start_index+4:question_end_index+1]


    final_output(context = context_str, question=question_str)


test('운영체제는 파일 시스템 관리, 프로세스 스케줄링, 메모리 관리, 동기화 및 상호배제 등의 다양한 기능을 수행한다.')
