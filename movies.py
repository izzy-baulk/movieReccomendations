##for col in data.columns:
 #   print(col)

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from flask import Flask, render_template, request
import string, random
import movieposters as mp

app = Flask(  # Create a flask app
	__name__,
	template_folder='templates',  # Name of html file folder
	static_folder='static'  # Name of directory for static files
)

ok_chars = string.ascii_letters + string.digits

@app.route('/')
def form():
  return render_template('form.html')

#import and merge movie data
df1=pd.read_csv('Data/tmdb_5000_credits.csv')
df2=pd.read_csv('Data/tmdb_5000_movies.csv')
df1.columns = ['id','title','cast','crew']
df2= df2.merge(df1,on='id')
df2.rename(columns = {'title_x':'title'}, inplace = True)
del df2['title_y']

#create weighted score
C = df2['vote_average'].mean() #mean number of votes on iMDB
m = df2['vote_count'].quantile(0.9) #minimu votes needed (90th percentile)
#find movies that satisfy this
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape

def weighted_rating(x, m=m, C=C):
  v = x['vote_count']
  R = x['vote_average']
  # Calculation based on the IMDB formula
  return (v/(v+m) * R) + (m/(m+v) * C)

#add score column to table and sort
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

def clean_data(x):
  if isinstance(x, list):
    return [str.lower(i.replace(' ', '')) for i in x]
  else:
    if isinstance(x, str):
      return str.lower(x.replace(' ', ''))
    else:
      return ''

#filter by content (term frequency - inverse document frequency)
tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('') #clean data
tfidf_matrix = tfidf.fit_transform(df2['overview']) #create matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) #cosin similarity
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates() #be able to get index from title

def get_recommendations(title, cosine_sim=cosine_sim):
  idx = indices[title]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:11] #ignore first movie as that is the movie itself
  movie_indices = [i[0] for i in sim_scores]
  films = df2['title'].iloc[movie_indices] #makes a series
  films = films.to_string(header=False, index=False).split('\n')
  #films = [','.join(ele.split()) for ele in films]
  return films

def get_director(x):
  for i in x:
    if i['job'] == 'Director':
      return i['name']
  return np.nan

def get_list(x):
  if isinstance(x, list):
    names = [i['name'] for i in x]
  if len(names) > 3: #get top 3 actors if more than 3
    names = names[:3]
  return names
  return []

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
  df2[feature] = df2[feature].apply(literal_eval)
#define new features
df2['director'] = df2['crew'].apply(get_director)
for feature in features:
  df2[feature] = df2[feature].apply(get_list)
  df2[feature] = df2[feature].apply(clean_data)

#create metdata soup used to get reccomendations based on cast and crew
def create_soup(x):
  return ' '.join(str(x['keywords'])) + ' ' + ' '.join(str(x['cast'])) + ' ' + str(x['director']) + ' ' + ' ' + ' '.join(str(x['genres']))
df2['soup'] = df2.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
  
@app.route('/data', methods = ['POST', 'GET'])
def data():
  if request.method == 'POST':
    form_data = request.form.get('Film')
    films = get_recommendations(form_data, cosine_sim2)
    posters = []
    return render_template('index.html',films=films, film=form_data)

  
if __name__ == "__main__":  # Makes sure this is the main process
	app.run( # Starts the site
		host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
		port=random.randint(2000, 9000)  # Randomly select the port the machine hosts on.
	)
