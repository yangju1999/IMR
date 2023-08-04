from datasets import load_dataset

data = load_dataset("squad_kor_v1")

'''
data = data.map(
    lambda x: {'text': f"### 문맥: {x['context']}\n\n### 질문: {x['question']}\n\n### 답변: {x['answers']['text'][0]}<|endoftext|>" }
)
'''

data = data.map(
    lambda x: {'text': f"### 문맥: {x['context']}\n\n### 질문: {x['question']}<|endoftext|>" }
)



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "EleutherAI/polyglot-ko-5.8b"  # safetensors 컨버팅된 레포
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)




from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


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


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)



import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=30000, ## 초소량만 학습: 50 step만 학습. 약 4분정도 걸립니다.  train data가 총 6만개, batch size가 2이므로 3만개로 설정할때 1epoch
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

def gen(x):
    gened = model.generate(
        **tokenizer(
            f"### 문맥: {x}\n\n### 질문:",
            return_tensors='pt',
            return_token_type_ids=False
        ),
        max_new_tokens=256,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))

gen("상당 부분 복구된 동태평양 개체군을 중심으로 이들의 회유지 경로인 브리티시 컬럼비아, 워싱턴 주, 캘리포니아에 범고래 등을 묶어 함께 관찰하는 관광 사업이 많이 발달되어 있다. 회유 기간에는 고래 관광에 가장 적합한 고래 중의 하나로 꼽힌다. 과거에 포경업자에게 보인 사나운 태도와는 달리 관찰자에게는 호기심 깊게 접근하며, 심지어 머리를 쓰다듬게 놔두기도 한다. 이러한 귀신고래가 인간에게 보이는 친근감은 1970년대부터 높아지고 있으며, 관찰선에 몸을 비비는 것 또한 관찰되었다. 이들을 관찰하기에 좋은 달은 1월에서 3월 사이에 남하할 때이며, 북상할 때는 해안에 거리를 두고 이동하기 때문에 최적기로 여겨지지 않는다."	
)