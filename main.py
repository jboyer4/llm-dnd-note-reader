from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Read the text from myData.txt
with open("myData.txt", "r", encoding="utf-8") as myData:
    text = myData.read()


# create embeddings with GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = "[PAD]"
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# load the pre-trained model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate embeddings using the model
with torch.no_grad():
    outputs = model(**encoded_input)
    embeddings = outputs[0]

# prepare a question to ask the model
question = "What were the main outcomes of world war 2?"

# encode the question
encoded_question = tokenizer(
    question, padding=True, truncation=True, return_tensors="pt"
)

# Extract tensors from BatchEncoding object
question_tensor = encoded_question["input_ids"]
embeddings_tensor = embeddings[0]
print(embeddings_tensor.shape)
print(question_tensor.shape)
print("With unsqueeze")
print(embeddings_tensor.unsqueeze(0).shape)
print(question_tensor.unsqueeze(2).shape)
# Concatenate the question encoding with the embeddings
# Using unsqueeze to add a dimension to the tensor so that they can be concatenated along that dimension
input_ids = torch.cat(
    [question_tensor.unsqueeze(2), embeddings_tensor.unsqueeze(0)], dim=2
)

# Generate the answer using the GPT model
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=100)

# Decode and print the generated answer
answer = tokenizer.decode(
    outputs[:, encoded_question.shape[1] :][0], skip_special_tokens=True
)
print("Answer:", answer)
