from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Doc1 = 'book book music video video'
Doc2 = 'music music video'
Doc3 = 'book book video'

docs = [Doc1, Doc2, Doc3]
def cosine_sim_function(count_vec, count_vec_name):
    for i in range(3):
        unigram_count = count_vec(encoding='latin-1')
        vec = unigram_count.fit_transform(docs)
        cos_sim = cosine_similarity(vec[i], vec)
        print(count_vec_name)
        print(cos_sim, '\n')

cosine_sim_function(CountVectorizer, 'Base Count Vectorizer')
cosine_sim_function(TfidfVectorizer, 'TFIDF Vectorizer')
