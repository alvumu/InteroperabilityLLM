#!pip install -U sentence-transformers
 
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Sample sentence
sentences = ["The movie is awesome. It was a good thriller",
             "We are learning NLP throughg GeeksforGeeks",
             "The baby learned to walk in the 5th month itself"]
 
 
test = "I liked the movie."
print('Test sentence:',test)
test_vec = model.encode([test])[0]
 
 
for sent in sentences:
    similarity_score = 1-distance.cosine(test_vec, model.encode([sent])[0])
    print(f'\nFor {sent}\nSimilarity Score = {similarity_score} ')