from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os, time



# path = "corpuses"
path = "linecorpus"
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
files = files[:10]
# files = [f"wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
print(len(files))
print(files[:10])


t1 = time.time()
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])


tokenizer.train(files, trainer)
tokenizer.save("tokenizer-wiki.json")
# tokenizer.save("path-tokenizer-l.json")
t2 = time.time()
print(f"Time(s) elapsed: {t2 - t1}")
# tokenizer = Tokenizer.from_file("path-tokenizer.json")
