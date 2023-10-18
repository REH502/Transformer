from hyperparams import Hyperparams as hp
import codecs
import os
import regex



def make_vocab(fpath, fname):
    """Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    """
    text = codecs.open(fpath, "r", "utf-8").read()
    text = regex.sub("[^\s\p{L}']", "", text)
    words = text.split()
    words = set(words)
    word_index = {w: i for i, w in enumerate(words)}
    if not os.path.exists("data"):
        os.mkdir("data")
    with codecs.open("data/{}".format(fname), "w", "utf-8") as fout:
        fout.write(
            "{}\t 0\n{}\t3\n{}\t1\n{}\t2\n".format(
                "<PAD>", "<UNK>", "<S>", "</S>"
            )
        )
        for word, cnt in word_index.items():
            fout.write(u"{}\t{}\n".format(word, cnt + 4))


if __name__ == "__main__":
    make_vocab(hp.source_train, "cn.txt.vocab.tsv")
    make_vocab(hp.target_train, "en.txt.vocab.tsv")
    print("Done")