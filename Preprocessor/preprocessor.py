import urllib
import html
import re

from urlextract import URLExtract
from unicodedata import normalize

from .demojize import demojize


def hashtag_handler(text: str):
    pattern = r"(#([^\s]+))"
    return re.sub(pattern, " <hashtag> \\2 </hashtag> ", text)


def cashtag_handler(text: str):
    pattern = r"(\$([^\s]+))"
    return re.sub(pattern, " <cashtag> \\2 </cashtag> ", text)


def mention_handler(text: str):
    pattern = r"(@([^\s]+))"
    return re.sub(pattern, " @user ", text)


url_extractor = URLExtract()


def url_handler(text: str):
    urls = list(url_extractor.gen_urls(text))
    updated_urls = list(
        set([url if "http" in url else f"https://{url}" for url in urls])
    )
    domains = [urllib.parse.urlparse(url_text).netloc for url_text in updated_urls]
    for i in range(len(domains)):
        text = text.replace(urls[i], f" <http> {domains[i]} </http> ")
    return text


def email_handler(text: str):
    pattern = r"[\w.+-]+@[\w-]+\.[\w.-]+"
    match = re.findall(pattern, text)
    for m in match:
        text = text.replace(m, " <email> ").strip()
    return text


def emoji_handler(text: str):
    return demojize(text, language="tr", delimiters=(" <emoji> ", " </emoji> "))


def normalize_text(text: str):
    return normalize("NFC", text)


def preprocess(text: str):
    output = html.unescape(text)
    output = normalize_text(output)
    output = email_handler(output)
    output = url_handler(output)
    output = hashtag_handler(output)
    output = cashtag_handler(output)
    output = mention_handler(output)
    output = emoji_handler(output)
    output = re.sub(r"\s+", " ", output)
    output = output.lower()
    output = output.strip()

    return output


if __name__ == "__main__":
    sample_text = ""
    preprocessed_text = preprocess(sample_text)
    print(preprocessed_text)
