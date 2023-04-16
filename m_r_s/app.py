from flask import Flask,render_template,request
import pickle
import numpy as np

pop_df=pickle.load(open('pop.pkl','rb'))
pt=pickle.load(open('pt.pkl','rb'))
model_knn=pickle.load(open('model_knn.pkl','rb'))
mr=pickle.load(open('mr.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
                           title=list(pop_df['title'].values),
                           avg=list(pop_df['avg_rating'].values),
                           genre=list(pop_df['genres'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_movies',methods=['POST'])
def recommend():
    user_input=request.form.get('user_input')
    index=np.where(pt.index==user_input)[0][0]
    distances,indices=model_knn.kneighbors(pt.iloc[index,:].values.reshape(1,-1),n_neighbors=11)
    movie=[]
    movgen=[]
    for i in range(0,len(distances.flatten())):
        if(i!=0):
            movie.append(pt.index[indices.flatten()[i]])
    for i in movie:
        movgen_t=[]
        temp_df=mr[mr['title']==i]
        movgen_t.append(i)
        movgen_t.extend(list(temp_df.drop_duplicates('title')['genres'].values))
        movgen.append(movgen_t)
    print(movgen)
    return render_template('recommend.html',movgen=movgen)

if __name__=='__main__':
    app.run(debug=True)