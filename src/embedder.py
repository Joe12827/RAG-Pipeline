from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm

class Qwen3Embedder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states, attention_mask):
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(seq_lens.size(0)), seq_lens]

    def embed(self, inputs, already_tokenized=False):
        # Tokenize only if not already tokenized
        if not already_tokenized:
            inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        pooled = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(pooled, p=2, dim=1)
        return embeddings.cpu().numpy()

