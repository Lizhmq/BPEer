# BPEer

Train BPE with fastBPE, and load to Huggingface Tokenizer.

### Description

The [BPETrainer](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html) of Huggingface consumes a lot of memory when I am training on a large corpus (e.g. 50000 merges on 20GB corpus). And I got a memory error.

So I use [fastBPE](https://github.com/glample/fastBPE) (implemented with C) instead, which returns a list of merge operations.

However, I still want to use the huggingface Tokenizer API. So I write a simple convertor for generating the json file for Huggingface Tokenizer.

### Usage

Train BPE:
```bash
cd fastBPE
./fast learnbpe [merges, e.g. 50000] [train.txt] > allvocab
```
Convert to json:
```bash
python convertjs.py
```

### Warning

This tokenizer does not indicate the start of a token.

E.g. BPE result for "I am" and "Iam" may be the same. Please split the sentence by space before you use it.
```python
    words = "I am".split()
    for word in words:
        subs = tokenizer.tokenize(word)
        subs[0] = "<begin>" + subs[0]
```
This results in ["\<begin\>I", "am"] and ["\<begin\>I", "\<begin\>am"] for "Iam" and "I am".