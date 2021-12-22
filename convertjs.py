import json
from tokenizers import Tokenizer
from tokenizers.models import BPE


def read_vocab(path):
    ret = []
    with open(path, "r") as f:
        for line in f:
            x, y = line.strip().split()[:2]
            x, y = x.replace("</w>", ""), y.replace("</w>", "")
            ret.append((x, y))
    return ret


def convert():
    infile = "fastBPE/allvocab"
    # infile = "testo.merge"
    outfile = "vocab.json"
    tmpvocab = json.load(open("template.json"))
    curidx = len(tmpvocab["model"]["vocab"])
    merges = read_vocab(infile)
    init_vocab = []
    lst = sorted(list(set("".join(x + y for x, y in merges))))
    init_vocab.extend(lst)
    for v in init_vocab:
        tmpvocab["model"]["vocab"][v] = curidx
        curidx += 1
    for a, b in merges:
        try:
            assert a in init_vocab
            assert b in init_vocab
        except AssertionError:
            print(a, b)
        init_vocab.append((a + b))
    for a, b in merges:
        tmpvocab["model"]["merges"].append(a + " " + b)
        tmpvocab["model"]["vocab"][a + b] = curidx
        curidx += 1

    t = json.dump(tmpvocab, open(outfile, "w"), indent=4)


def test():
    tokenizer = Tokenizer.from_file("pathvocab.json")
    t = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|. program|"
    ).tokens
    id = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|. program|"
    ).ids
    off = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|. program|"
    ).offsets
    print(t)
    print(id)
    # return


if __name__ == "__main__":
    convert()
    test()
