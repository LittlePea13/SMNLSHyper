"""
Classes representing "processing resources"
"""

import xml.etree.ElementTree
from collections import Counter
import json
from numbers import Number
from html.parser import HTMLParser

import sys
import xml.etree.ElementTree

import re

txt_tospace1 = re.compile('&#160;')
FEATURES = [
]


def doc2features(doc, features):
    """
    Extract the features from the document. Each feature is either just the name
    or a tuple of (name,flag,type) where flag indicates if the feature should get
    selected. This returns the features in their original representation.
    :param doc: to extract the features from
    :param features: list of feature names or 3-tuples
    :return: list of values for the selected features
    """
    ret = []
    for f in features:
        if isinstance(f, str):
            val = doc.get(f)
            ret.append(val)
        else:
            name, flag, ftype = f
            if flag:
                val = doc.get(name)
                ret.append(val)
    return ret

def features2use(features):
    """
    Return just those features which have flag true or are just the name, not the tuple
    :param features:
    :return:
    """
    ret = []
    for f in features:
        if isinstance(f, str):
            ret.append(f)
        else:
            name, flag, ftype = f
            if flag:
                ret.append(f)
    return ret

def cleantext(text):
    '''Clean the text extracted from XML.'''
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("<p>", " ")
    text = text.replace("</p>", " ")
    text = text.replace(" _", " ")
    text = text.replace("–", "-")
    text = text.replace("”", "\"")
    text = text.replace("“", "\"")
    text = text.replace("’", "'")

    text, _ = txt_tospace1.subn(' ', text)
    return text

class PrArticle2Line:

    def __init__(self, stream, featureslist, addtargets=True):
        self.stream = stream
        self.features = features2use(featureslist)
        self.mp_able = False
        self.addtargets = addtargets
        self.need_et = False

    def __call__(self, article, **kwargs):
        values = doc2features(article, self.features)
        strings = []
        for i in range(len(values)):
            val = values[i]
            if isinstance(val, str):
                strings.append(val)
            elif isinstance(val, Number):
                strings.append(str(val))
            elif isinstance(val, list):
                strings.append(json.dumps(val))
            elif isinstance(val, dict):
                strings.append(json.dumps(val))
            else:
                # raise Exception("Not a known type to convert to string: {} for {}, feature {}, article id {}".
                # format(type(val), val, self.features[i], article['id']))
                print("Not a known type to convert to string: {} for {}, feature {}, article id {}".
                      format(type(val), val, self.features[i], article['id']))
        if self.addtargets:
            print(article['id'], article.get('target'),
                  article.get('bias'), article.get('domain'),
                  "\t".join(strings), file=self.stream, sep="\t")
        else:
            print("\t".join(strings), file=self.stream)

class PrText2Line:

    def __init__(self, stream, featureslist):
        self.stream = stream
        self.mp_able = False
        self.need_et = False
        self.features = featureslist
    def __call__(self, article, **kwargs):
        values = doc2features(article, self.features)
        strings = text
        print(text)
        print("\t".join(strings), file=self.stream)


class PrAddTarget:

    def __init__(self, a2target, a2bias, a2url):
        self.a2target = a2target
        self.a2bias = a2bias
        self.a2url = a2url
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        id = article['id']
        target = self.a2target[id]
        bias = self.a2bias[id]
        url = self.a2url[id]
        article['target'] = target
        article['bias'] = bias
        article['url'] = url


class PrAddTitle:

    def __init__(self):
        self.mp_able = True
        self.need_et = True

    def __call__(self, article, **kwargs):
        element = article['et']
        attrs = element.attrib
        title = cleantext(attrs["title"])
        article['title'] = title

class MyHTMLParser(HTMLParser):

    def __init__(self):
        kwargs = {}
        HTMLParser.__init__(self, **kwargs)
        self.ignore = False
        self.data = []
        self.p = []

    def finishp(self):
        if len(self.p) > 0:
            self.data.append(self.p)
            self.p = []

    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        if tag in ['script', 'style']:
            self.ignore = True
        elif tag in ['p', 'br']:
            self.finishp()
        # any tags that need to get repalced by space?
        # elif tag in ['???']:
        #     self.p.append(" ")

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        if tag in ['script', 'style']:
            self.ignore = False
        elif tag in ['p', 'br']:
            self.finishp()
        # any tags that need to get repalced by space?
        # elif tag in ['???']:
        #     self.p.append(" ")

    def handle_startendtag(self, tag, attrs):
        # print("Encountered a startend tag:", tag)
        if tag in ['p', 'br']:
            self.finishp()

    def handle_data(self, data):
        # print("Encountered some data  :", data)
        if not self.ignore:
            self.p.append(data)

    def close(self):
        HTMLParser.close(self)
        self.finishp()

    def reset(self):
        HTMLParser.reset(self)
        self.data = []
        self.p = []

    def cleanparagraph(self, text):
        """
        How to do basic cleaning up of the text in each paragraph
        :return:
        """
        text = cleantext(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = ' '.join(text.split()).strip()
        return text

    def paragraphs(self):
        """
        Convert collected data to paragraphs
        """
        pars = []
        for par in self.data:
            if len(par) > 0:
                text = self.cleanparagraph(''.join(par)).strip()
                if text:
                    pars.append(text)
        return pars


class PrAddText:

    def __init__(self):
        self.mp_able = True
        self.need_et = False
        self.parser = None  # initialize later, do not want to pickle for the pipeline

    def __call__(self, article, **kwargs):
        if self.parser is None:
            self.parser = MyHTMLParser()
        self.parser.reset()
        self.parser.feed(article['xml'])
        self.parser.close()
        pars = self.parser.paragraphs()
        article['pars'] = pars
        print(pars)
        text = " ".join(pars)
        print(text)
        article['text'] = text


class PrRemovePars:

    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        del article['pars']


class PrFilteredText:
    """
    Calculate the single filtered text field text_all_filtered, must already have nlp
    """

    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        import nlp
        text_tokens = article['text_tokens']
        title_tokens = article['title_tokens']
        tokens = nlp.filter_tokens([t[0] for t in title_tokens])
        tokens.append("<sep_t2d>")
        if article.get('link_domains_all'):
            tokens.extend(["DOMAIN_" + d for d in article['link_domains']])
        tokens.append("<sep_d2a>")
        tokens.extend(nlp.filter_tokens([t[0] for sent in text_tokens for t in sent]))
        token_string = " ".join(tokens)
        article['text_all_filtered'] = token_string

class PrFilteredTextAtt:
    """
    Calculate the single filtered text field text_all_filtered, must already have nlp
    """

    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        import nlp
        tokens = nlp.filter_tokens(article)
        tokens.append("<sep_t2d>")
        tokens.append("<sep_d2a>")
        tokens.extend(nlp.filter_tokens([t[0] for sent in text_tokens for t in sent]))
        token_string = " ".join(tokens)
        article['text_all_filtered'] = token_string


class PrNlpSpacy01:
    """
    Tokenise and POS-tag the title and article.
    The title gets converted into a list of list word, POS, lemma.
    The article gets converted into a list of
    sentences containing a list of lists word, POS, lemma for the sentence.
    :return:
    """

    def __init__(self):
        import spacy
        self.mp_able = True
        self.initialized = False
        self.need_et = False
        self.nlp = None

    def initialize(self):
        if self.initialized:
            return
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.initialized = True

    def __call__(self, article, **kwargs):
        # process each paragraph separately to avoid getting sentences
        # crossing paragraphs
        if not self.initialized:
            self.initialize()
        pars = article['pars']
        # store the raw number of paragraphs
        article['n_p'] = len(pars)
        #print("DEBUG: number of pars", len(pars))
        n_p_filled = 0
        #print("\n\nDEBUG: {} the texts we get from the paragraphs: ".format(article['id']), pars)
        docs = list(self.nlp.pipe(pars))
        for doc in docs:
            doc.is_parsed = True
        allthree = [[[t.text, t.pos_, t.lemma_] for t in s] for doc in docs for s in doc.sents]
        article['n_p_filled'] = n_p_filled
        article['text_tokens'] = allthree
        ents = [ent.text for doc in docs for ent in doc.ents if ent.text[0].isupper()]
        article['text_ents'] = ents
        title = article['title']
        doc = self.nlp(title)
        doc.is_parsed = True
        allthree = [(t.text, t.pos_, t.lemma_) for s in list(doc.sents) for t in s]
        article['title_tokens'] = allthree
        ents = [ent.text for ent in doc.ents if ent.text[0].isupper()]
        article['title_ents'] = ents


class PrSeqSentences:
    """
    Creates fields: title_sent, domain_sent, article_sent, title and article generated from the
    token lists for the title and article text (using the original token string)
    The sentences for the article are enclosed in the special <bos> and <eos> markers.
    """
    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        article_tokens = article['text_tokens']
        title_tokens = article['title_tokens']
        title_sent = " ".join([t[0] for t in title_tokens])
        domain_sent = ""
        if article.get('link_domains_all'):
            domain_sent = " ".join(["DOMAIN_" + d for d in article['link_domains']])
        all = []
        first = True
        for sent in article_tokens:
            if first:
                first = False
            else:
                all.append("<splt>")
            # all.append("<bos>")
            for t in sent:
                all.append(t[0])
            # all.append("<eos>")
        article_sent = " ".join(all)
        article['article_sent'] = article_sent
        article['domain_sent'] = domain_sent
        article['title_sent'] = title_sent


class PrNlpSpacy01Att:
    """
    Tokenise and POS-tag the title and article.
    The title gets converted into a list of list word, POS, lemma.
    The article gets converted into a list of
    sentences containing a list of lists word, POS, lemma for the sentence.
    :return:
    """

    def __init__(self):
        import spacy
        self.mp_able = True
        self.initialized = False
        self.need_et = False
        self.nlp = None

    def initialize(self):
        if self.initialized:
            return
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.initialized = True

    def __call__(self, article, **kwargs):
        # process each paragraph separately to avoid getting sentences
        # crossing paragraphs
        if not self.initialized:
            self.initialize()
        pars = article['pars']
        # store the raw number of paragraphs
        article['n_p'] = len(pars)
        #print("DEBUG: number of pars", len(pars))
        n_p_filled = 0
        #print("\n\nDEBUG: {} the texts we get from the paragraphs: ".format(article['id']), pars)
        docs = list(self.nlp.pipe(pars))
        for doc in docs:
            doc.is_parsed = True
        allthree = [[[t.text, t.pos_, t.lemma_] for t in s] for doc in docs for s in doc.sents]
        article['n_p_filled'] = n_p_filled
        article['text_tokens'] = allthree
        ents = [ent.text for doc in docs for ent in doc.ents if ent.text[0].isupper()]
        article['text_ents'] = ents


class PrSeqSentencesAtt:
    """
    Creates fields: title_sent, domain_sent, article_sent, title and article generated from the
    token lists for the title and article text (using the original token string)
    The sentences for the article are enclosed in the special <bos> and <eos> markers.
    """
    def __init__(self):
        self.mp_able = True
        self.need_et = False

    def __call__(self, article, **kwargs):
        article_tokens = article['text_tokens']
        all = []
        first = True
        for sent in article_tokens:
            if first:
                first = False
            else:
                all.append("<splt>")
            # all.append("<bos>")
            for t in sent:
                all.append(t[0])
            # all.append("<eos>")
        article_sent = " ".join(all)
        article['article_sent'] = article_sent
