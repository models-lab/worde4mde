from pprint import pprint

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("roberta-modeling/checkpoint-38550", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer, device_map="auto")
text = "Thus, it is easy to implement the formal <mask> of type system and operational semantics"

pprint(mask_filler(text, top_k=5))
