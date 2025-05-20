from transformers import BertTokenizer, BertForSequenceClassification

# Укажи путь для сохранения
save_path = "toxic_bert_model"

# Загружаем модель и токенизатор
model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")

# Сохраняем локально
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Модель и токенизатор сохранены в папку: {save_path}")
