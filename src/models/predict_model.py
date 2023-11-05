from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = '../../models/bart_seq2seq'
device = 'cpu'

model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def detoxify(text: str):
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
