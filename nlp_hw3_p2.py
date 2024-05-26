import json

from datasets import Dataset

import torch
import accelerate

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def EncodeInstance(instance):
  fact, stem = instance['fact1'], instance['question']['stem']
  choices = ' '.join([f"[{choice['label']}] {choice['text']}" for choice in instance['question']['choices']])
  answer = instance['answerKey']

  prompt = f"[START] {fact} {stem} {choices} [ANSWER] {answer}"
  return prompt

def TokenizeInstance(instance, tokenizer):
    encoded = tokenizer(instance,
                        padding = "max_length",
                        truncation = True,
                        max_length = 512,
                        return_tensors = 'pt')

    labels = encoded['input_ids'].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': encoded['input_ids'],'attention_mask':encoded['attention_mask'],'labels':labels}

    


def LoadDataAndTokenize(data_path, tokenizer):
  with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f] 


  prompts = [EncodeInstance(instance) for instance in data]
  tokenized_data = [TokenizeInstance(prompt, tokenizer) for prompt in prompts]

  input_ids = torch.cat([item['input_ids'] for item in tokenized_data], dim = 0)
  attention_mask = torch.cat([item['attention_mask'] for item in tokenized_data], dim = 0)
  labels = torch.cat([item['labels'] for item in tokenized_data], dim = 0)

  dataset = Dataset.from_dict({
      'input_ids':input_ids,
      'attention_mask': attention_mask,
      'labels': labels
  })

  return dataset


class CustomGPT2(GPT2LMHeadModel):
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels)
        if labels is not None:
            loss = outputs.loss
            return {'loss': loss, 'logits': outputs.logits}
        return outputs

 if __name__ == "__main__":
 	train_path = r"/train_complete.jsonl"
	test_path = r"/test_complete.jsonl"

	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
	tokenizer.pad_token = tokenizer.eos_token

	train_data, test_data = LoadDataAndTokenize(train_path, tokenizer), LoadDataAndTokenize(test_path, tokenizer)

	model = CustomGPT2.from_pretrained('gpt2')

	training_args = TrainingArguments(
						    output_dir='./results',
						    overwrite_output_dir=True,
						    num_train_epochs=3,
						    per_device_train_batch_size=2,
						    save_steps=10_000,
						    save_total_limit=2,
						    logging_dir='./logs',
						    logging_steps=200)

	trainer = Trainer(
		    model=model,
		    args=training_args,
		    train_dataset=train_data,
		    tokenizer=tokenizer)

	trainer.train()

	predictions = trainer.predict(test_data)

	decoded_predictions = []
	for input_id in predictions.predictions:
	    decoded_predictions.append(tokenizer.decode(input_id, skip_special_tokens=True))

	for i, prediction in enumerate(decoded_predictions[:10]):
	    print(f"Example {i + 1}: {prediction}")


