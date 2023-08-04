from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


# Initialize instance of FastAPI
app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#model 및 tokenizer load하기 
peft_model_id = "./outputs/checkpoint-500"  #finetuned 모델 path  
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

def get_answer(prompt):
    #prompt = """Return True if the given article is fake. article: Boeing CEO says he assured Trump about Air Force One costs answer:"""
    q = f"### 질문: {prompt}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=200,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    output = tokenizer.decode(gened[0])
    output = output[len(prompt)+18:]
    return(output)

@app.get('/')
async def root():
    return {'message': 'Hello World'}


@app.get('/chat_test')
async def test(user_message):
    return {'message': get_answer(user_message)}


@app.post('/chat')
async def chat(param: dict={}):
   user_message = param.get('user_message', ' ')
   return {'message': get_answer(user_message)}