inelmo="$1"
outpreds="$2"
#inelmo='Data/Hyperpartisan/articles-training-byarticle-20181122.xml'
/anaconda3/envs/statnlp/bin/python3 Data/Hyperpartisan/test_parser.py -A $inelmo -F article_sent,title_sent test.text.tsv
/anaconda3/envs/statnlp/bin/python3 Data/elmo_power.py -TSV test.text.tsv
#/anaconda3/envs/statnlp/bin/python3 test_model.py -out $outpreds -model Model/hyper_model_15epoch.pt