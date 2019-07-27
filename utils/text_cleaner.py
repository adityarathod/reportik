import re
from nltk.tokenize import sent_tokenize

text_replacements = {
        'U. S.': 'US',
        'U. K.': 'UK',
        'Sen.': 'Sen'
}


def is_hidden_file(name: str):
    return name.startswith('.')


def clean_sentence(sentence: str):
    out = sentence[:-1]
    for orig, new in text_replacements.items():
        out = out.replace(orig, new)
    return out + ' <PUNCT> '


def clean_text(txt: str):
    fixed_txt = re.sub(r'\.(?=[^ \W\d])', '. ', txt)
    fixed_txt = fixed_txt.replace('\n', ' <NEWLINE> ')
    sentences = sent_tokenize(fixed_txt)
    sentences = [clean_sentence(s.strip()) for s in sentences]
    return '<START> ' + ''.join(sentences) + '<EOS>'