# dataset load 
from datasets import load_dataset
import random 

#hugging face에서 squad_kor_v1 데이터 셋 가져오기 
data = load_dataset("squad_kor_v1")

# 오버피팅을 방지하기 위해 조금 더 다양한 프롬프트들을 정의함
prompts = ['위의 문맥을 읽고 풀 수 있는 문제 하나 만들어줘', '지금까지의 문맥 정보를 바탕으로 질문 하나 생성해줘', '위의 문맥 내용을 통해 맞출 수 있는 문제 한 개만 작성해줘', '위의 문맥에서 정답을 찾을 수 있는 질문 하나 생성해줘']

#4가지의 프롬프트 중 랜덤하게 하나를 골라서 문맥, 질문과 조합하여 적절한 학습용 문장을 생성 
data = data.map(
    lambda x: {'text': f"### 문맥: {x['context']}\n\n" + prompts[random.randrange(0,4)]+ f"### 질문: {x['question']}<|endoftext|>" }
)



# pretrain 모델 load
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "EleutherAI/polyglot-ko-5.8b"   # huggingface에서 pretrained 모델 가져오기 
bnb_config = BitsAndBytesConfig(    # 4 bit 양자화 사용을 위한 코드 
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)



# 학습에서 peft 방식을 사용하기 위한 코드 
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# peft 방식을 사용했을 때, update하는 parameter 개수를 확인하기 위한 함수
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# peft 방식 중에서 Lora 를 사용하기 위한 코드
from peft import LoraConfig, get_peft_model

config = LoraConfig(     # r 값으로 얼마만큼의 parameter 들을 학습할 것인 지 조절 
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



#실제 train 하는 부분 
import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

#huggingface에서 제공하는 Trainer 모듈을 사용하여 hyperparamer 만 넣어주면 쉽게 학습 가능 
trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1000, ## 초소량만 학습: train data가 총 6만개, batch size가 2이므로 3만개로 설정할때 1 epoch
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs_qg",
        optim="paged_adamw_8bit"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.eval()
model.config.use_cache = True


#학습 이후 가볍게 validation 해보기 위한 함수  
def gen(x):
    gened = model.generate(
        **tokenizer(
            f"### 문맥: {x}\n\n위의 문맥에서 정답을 찾을 수 있는 질문 하나 생성해줘### 질문:",
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    result = tokenizer.decode(gened[0])
    question_start_index = result.find('질문:')
    question_end_index = question_start_index + result[question_start_index:].find('?')
    print(result[question_start_index+3: question_end_index+1])

gen("상당 부분 복구된 동태평양 개체군을 중심으로 이들의 회유지 경로인 브리티시 컬럼비아, 워싱턴 주, 캘리포니아에 범고래 등을 묶어 함께 관찰하는 관광 사업이 많이 발달되어 있다. 회유 기간에는 고래 관광에 가장 적합한 고래 중의 하나로 꼽힌다. 과거에 포경업자에게 보인 사나운 태도와는 달리 관찰자에게는 호기심 깊게 접근하며, 심지어 머리를 쓰다듬게 놔두기도 한다. 이러한 귀신고래가 인간에게 보이는 친근감은 1970년대부터 높아지고 있으며, 관찰선에 몸을 비비는 것 또한 관찰되었다. 이들을 관찰하기에 좋은 달은 1월에서 3월 사이에 남하할 때이며, 북상할 때는 해안에 거리를 두고 이동하기 때문에 최적기로 여겨지지 않는다."	
)