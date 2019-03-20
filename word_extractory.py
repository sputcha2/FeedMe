from newsplease import NewsPlease
from googlesearch import search_news
import time

new_list = ['en.wikipedia.org', 'twitter.com']
'''
def article_generator(keyword_query, num_articles):
    new_dict = {}
    for url in search_news(str(keyword_query), num=1, stop=num_articles):
        article = NewsPlease.from_url(str(url))
        if (article.text != None):
            #print(article.source_domain)
            if article.source_domain not in new_list:
                new_dict[article] = len(article.text)
    return new_dict
'''
def article_generator_text(keyword_query, num_articles):
    text = ''
    for url in search_news(str(keyword_query), num=1, stop=num_articles):
        article = NewsPlease.from_url(str(url))
        if (article.text != None):
            if article.source_domain not in new_list:
                text = text + article.text
    return text

