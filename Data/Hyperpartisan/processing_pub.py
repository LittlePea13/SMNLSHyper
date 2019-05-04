'''
Process the XML articles with a pipeline, extract a list of features and write
the features to one line per document in an output tsv file.
'''
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

import argparse
import utils, processingresources, features
default_F = "(hyperpartisan,)"
args = Namespace(
    A='articles-training-bypublisher-20181122.xml',
    T='ground-truth-training-bypublisher-20181122.xml',
    F='article_sent,title_sent',
    outfile = 'output_pub.tsv'
)
#parser = argparse.ArgumentParser()
#parser.add_argument("-A", default='articles-training-byarticle-20181122', type=str, required=True, help="Article XML file")
#parser.add_argument("-T", default='ground-truth-training-byarticle-20181122', type=str, help="Targets XML file, if missing, targets etc are None")
#parser.add_argument("-F", default=default_F, help="Feature list to use, or comma separated list of features ({})".format(default_F))
#parser.add_argument("outfile", type=str, help="Output (tsv) file")
#args = parser.parse_args()

features_string = args.F

if "," in features_string:
    tmp = features_string.split(",")
    features = [f for f in tmp if f]
else:
    features = getattr(features, features_string)

if args.T is None:
    a2target, a2bias, a2url = (None, None, None)
    pipeline = []
else:
    print("Loading targets")
    a2target, a2bias, a2url = utils.load_targets_file(args.T, cache=None)
    pipeline = [processingresources.PrAddTarget(a2target, a2bias, a2url)]

pipeline.extend([
    processingresources.PrAddTitle(),
    processingresources.PrAddText(),
    processingresources.PrNlpSpacy01(),
    processingresources.PrFilteredText(),
    processingresources.PrSeqSentences(),
    processingresources.PrRemovePars(),
])

with open(args.outfile, "wt", encoding="utf8") as outstream:
    pipeline.append(processingresources.PrArticle2Line(outstream, features, addtargets=True))
    print("Pipeline:")
    for p in pipeline:
        print(p)
    ntotal, nerror = utils.process_articles_xml(args.A, pipeline)
    print("Total processed articles:", ntotal)
    print("Errors (could be >1 per article):", nerror)