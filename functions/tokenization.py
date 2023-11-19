from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(), 
        add_special_tokens=True, 
        truncation=True, 
        padding=True, 
        return_tensors='pt'
    )
    
    