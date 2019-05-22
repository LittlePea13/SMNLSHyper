import argparse
import numpy as np
import subprocess
import h5py
def make_elmo_file(args):
    infile = args.TSV
    outfile = 'Att_article.txt'
    maxsent = 100
    hyp_elmo = []
    hyp_label = []
    hyp_len = []
    with open(infile, "rt", encoding="utf8") as inp:
        nlines = 0
        for enum,line in enumerate(inp):
            # a line will have 1 document
            fields = line.split("\t")
            label = fields[2]
            if(label == "true"):
                label = 1
            else:
                label = 0
            tmp = fields[4]
            tmp = tmp.split(" <splt> ")[:maxsent]
            number_of_sentences = len(tmp)
            hyp_len.append(number_of_sentences)
            sents = []
            sents.extend(tmp)
            sents = '\n'.join(sents)
            file = open(outfile, "w", encoding="utf8")
            file.write(sents)
            file.close()
            hyp_label.append(label)
    np.save('hyp_labels', hyp_elmo)
    try:
        subprocess.check_call(['allennlp elmo Att_article.txt elmo_hyp.hdf5 --average --weight-file Data/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 --options-file Data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'], shell=True)
    except subprocess.CalledProcessError:
        print('didnt work')
        pass # error
    except OSError:
        print('didnt work')
        pass # executable not found

    f = h5py.File("elmo_hyp.hdf5", 'r')

    # # List all groups

    a_group_key = list(f.keys())

    length = len(a_group_key)

    # Get the data
    final_embeds = []
    a = []
    for i in range(length - 1):
        a.append(str(i))
    counter = 0
    for i in a:
        data = list(f[i])
        final_embeds.append(data)
    all_docs = []
    counter = 0
    for document in hyp_len:
        document_emb = final_embeds[counter:counter+document]
        counter+=document
        all_docs.append(document_emb)
    #os.remove("elmo_hyp.hdf5")
    if(nlines % 100 == 0):
        print("Doing line", nlines)
    #break
    nlines += 1
    hyp_elmo.append(all_docs)
    #   print(len(hyp_elmo))
    print("Going to save files!")
    #np.save('hyp_labels', hyp_label)
    np.save('hyp_embeds', hyp_elmo)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-TSV", default='test.text.tsv', type=str, required=True, help="Article TSV file")
    args = parser.parse_args()
    make_elmo_file(args)