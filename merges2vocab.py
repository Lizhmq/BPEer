import json
from tokenizers import Tokenizer
from tokenizers.models import BPE


def read_vocab(path):
    ret = []
    with open(path, "r") as f:
        for line in f:
            x, y = line.strip().split()[:2]
            x, y = x.replace("</w>", "Ġ"), y.replace("</w>", "Ġ")
            # x, y = x[1:], y[:-2]
            ret.append((x, y))
    return ret


def convert():
    infile = "fastBPE/allvocab"
    # infile = "testo.merge"
    outfile = "pathvocab.json"
    tmpvocab = json.load(open("template.json"))
    curidx = len(tmpvocab["model"]["vocab"])
    merges = read_vocab(infile)
    init_vocab = []
    lst = sorted(list(set("".join(x + y for x, y in merges))))
    init_vocab.extend(lst)
    newmerges = []
    for c in lst:
        init_vocab.append(c + "Ġ")
        newmerges.append((c, "Ġ"))
    # merges = newmerges + merges
    for v in init_vocab:
        # if v not in tmpvocab["model"]["vocab"]:
        tmpvocab["model"]["vocab"][v] = curidx
        if curidx < 200:
            print(curidx)
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
        if curidx < 200:
            print(curidx)
        curidx += 1

    t = json.dump(tmpvocab, open(outfile, "w"), indent=4)


def test():
    tokenizer = Tokenizer.from_file("pathvocab.json")
    t = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|.Ġprogram|"
    ).tokens
    id = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|.Ġprogram|"
    ).ids
    off = tokenizer.encode(
        "program|block|if_statement|block|expression_statement|method_invocation|.Ġprogram|"
    ).offsets
    print(t)
    print(id)
    # return


if __name__ == "__main__":
    convert()
    test()
