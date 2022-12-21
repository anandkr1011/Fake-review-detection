# Sentiment Score Generation
reviews = data['Review_text'].tolist()
#print(reviews)
sentiment_score = []
sentiment_subjectivity=[]
review_head_sentiment=[]
for rev in reviews:
    testimonial = TextBlob(rev)
    sentiment_score.append(testimonial.sentiment.polarity)
    sentiment_subjectivity.append(testimonial.sentiment.subjectivity)
data['Sentiment'] = sentiment_score
data['Subjectivity'] = sentiment_subjectivity
data.head()


# Visualizing the sentiment

pos = 0
neg = 0
for score in data['Sentiment']:
    if score > 0:
        pos += 1
    elif score < 0:
        neg += 1

# Visualiing the distribution of Sentiment
values = [pos, neg]
label = ['Positive Reviews', 'Negative Reviews']

fig = plt.figure(figsize =(10, 7)) 
plt.pie(values, labels = label)
plt.show()


# Number of Negative words in a review


reviews = data['Review_text'].tolist()
negative_count = []
for rev in reviews:
    words = rev.split()
    neg = 0
    for w in words:
        testimonial = TextBlob(w)
        score = testimonial.sentiment.polarity
        if score < 0:
            neg += 1
    negative_count.append(neg)
data['Neg_Count'] = negative_count


# Unique words count
#Word Count

data['Word_Count'] = data['Review_text'].str.split().str.len()
for i in range(data.shape[0]):
    if data.loc[i].Word_Count == 0:
        data.drop(index=i, inplace=True)
data.reset_index(drop=True, inplace=True)
reviews = data['Review_text'].str.lower().str.split()

# Get amount of unique words
data['Unique_words'] = reviews.apply(set).apply(len)
#data['Unique_words'] = data[['Unique_words']].div(data.Word_Count, axis=0)


# POS tagging 

Review_text = data.Review_text

array_Noun = []
array_Adj = []
array_Verb = []
array_Adv = []
array_Pro = []
array_Pre = []
array_Con = []
array_Art = []
array_Nega = []
array_Aux = []

articles = ['a', 'an', 'the']
negations = ['no', 'not', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'barely', 'scarcely']
auxilliary = ['am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could', 'do', 'does', 'did', 'have', 'having', 'has', 'had']

for j in Review_text:
    text = j ;
    filter=re.sub('[^\w\s]', '', text)
    conver_lower=filter.lower()
    Tinput = conver_lower.split(" ")
    
    for i in range(0, len(Tinput)):
        Tinput[i] = "".join(Tinput[i])
    UniqW = Counter(Tinput)
    s = " ".join(UniqW.keys())
    
    tokenized = sent_tokenize(s)
    
    for i in tokenized:
        wordsList = nltk.word_tokenize(i)
        #wordsList = [w for w in wordsList if not w in stop_words]
        
        Art = 0
        Nega = 0
        Aux = 0
        for word in wordsList:
            if word in articles:
                Art += 1
            elif word in negations:
                Nega += 1
            elif word in auxilliary:
                Aux += 1
                
        tagged = nltk.pos_tag(wordsList)
        counts = Counter(tag for word,tag in tagged)

        N = sum([counts[i] for i in counts.keys() if 'NN' in i])
        Adj = sum([counts[i] for i in counts.keys() if 'JJ' in i])
        Verb = sum([counts[i] for i in counts.keys() if 'VB' in i])
        Adv = sum([counts[i] for i in counts.keys() if 'RB' in i])
        Pro = sum([counts[i] for i in counts.keys() if (('PRP' in i) or ('PRP$' in i) or ('WP' in i) or ('WP$' in i))])
        Pre = sum([counts[i] for i in counts.keys() if 'IN' in i])
        Con = sum([counts[i] for i in counts.keys() if 'CC' in i])

        array_Noun.append(N)
        array_Adj.append(Adj)
        array_Verb.append(Verb)
        array_Adv.append(Adv)
        array_Pro.append(Pro)
        array_Pre.append(Pre)
        array_Con.append(Con)
        array_Art.append(Art)
        array_Nega.append(Nega)
        array_Aux.append(Aux)
print('Completed')


POS = ['Noun_Count', 'Adj_Count', 'Verb_Count', 'Adv_Count', 'Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count', 'Nega_Count', 'Aux_Count']
Values = [array_Noun, array_Adj, array_Verb, array_Adv, array_Pro, array_Pre, array_Con, array_Art, array_Nega, array_Aux]
i = 0
for x in POS:
    data[x] = pd.Series(Values[i])
    data[x] = data[x].fillna(0)
    data[x] = data[x].astype(float)
    i += 1
    
 
# Authenticity
data = data.assign(Authenticity = lambda x: (x.Pro_Count + x.Unique_words - x.Neg_Count) / x.Word_Count)

# Analytical Thinking

data = data.assign(AT = lambda x: 30 + (x.Art_Count + x.Pre_Count - x.Pro_Count - x.Aux_Count - x.Con_Count - x.Adv_Count - x.Nega_Count))

# Labelling the Reviews

def label(Auth, At, N, Adj, V, Av, S, Sub, W):
    score = 0
    if Auth >= 0.49:
        score += 2
    if At <= 20:
        score += 1
    if (N + Adj) >= (V + Av):
        score += 1
    if -0.5 <= S <= 0.5:
        score += 1
    if Sub <= 0.5:
        score += 2
    if W > 75:
        score += 3
    if score >= 5:
        return 1
    else:
        return 0
data['Rev_Type'] = data.apply(lambda x: label(x['Authenticity'], x['AT'], x['Noun_Count'], x['Adj_Count'], x['Verb_Count'], x['Adv_Count'], x['Sentiment'], x['Subjectivity'], x['Word_Count']), axis = 1)
data['Rev_Type'].value_counts()
data.head()

# Model Training

df = data.loc[:, data.columns[4:-1]]
df.drop(['Review_text','Neg_Count','Unique_words','Pro_Count', 'Pre_Count', 'Con_Count', 'Art_Count',
       'Nega_Count', 'Aux_Count'], axis=1, inplace=True)
       

min_max_scaler = preprocessing.MinMaxScaler()
Columns=df.columns
df[Columns] = min_max_scaler.fit_transform(df[Columns])

x,y = df, data['Rev_Type']
RAN_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=RAN_STATE)


# HyperParameter Tuning

#Adaboost
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }

DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto", class_weight = "balanced",max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)

# run grid search
gs1 = GridSearchCV(
                    estimator=ABC, 
                    param_grid=param_grid, 
                    scoring = 'roc_auc')
print ('Fitting grid search...')
gs1.fit(X_train,y_train)
print ("Grid search fitted.")

# RandomForest

param_grid = {
'bootstrap': [True],
'max_depth': [80, 90, 100, 110],
'max_features': [2,3],
'min_samples_leaf': [2,3,4],
'min_samples_split': [2, 5, 10],
'n_estimators': [100, 200, 300, 1000]
}

gs2 = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=2),
    scoring='roc_auc',
    n_jobs = -1, 
    verbose = 2
)

print ('Fitting grid search...')
gs2.fit(X_train,y_train)
print ("Grid search fitted.")


#Logistic Regression
gs3 = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'C': [10**i for i in range(-5,5)], 'class_weight': [None, 'balanced']},
    cv=StratifiedKFold(n_splits=5),
    scoring='roc_auc'
)

#fit the grid search object to our new dataset

print ('Fitting grid search...')
gs3.fit(X_train, y_train)
print ("Grid search fitted.")

clf1 = gs1.best_estimator_
clf2 = gs2.best_estimator_
clf3 = gs3.best_estimator_
probas =clf1.predict(X_test)

# ROC/AUC score
print ('ROC_AUC Score:',accuracy_score(y_test, probas))


from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics

scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}
def models_evaluation(X, y, folds):
    aB = cross_validate(clf1, X, y, cv=folds, scoring=scoring)
    rF = cross_validate(clf2, X, y, cv=folds, scoring=scoring)
    lR = cross_validate(clf3, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'AdaBoost':[aB['test_accuracy'].mean(),
                                                               aB['test_precision'].mean(),
                                                               aB['test_recall'].mean(),
                                                               aB['test_f1_score'].mean()],

                                      'Random Forest':[rF['test_accuracy'].mean(),
                                                                   rF['test_precision'].mean(),
                                                                   rF['test_recall'].mean(),
                                                                   rF['test_f1_score'].mean()],

                                      'Logistic Regression':[lR['test_accuracy'].mean(),
                                                       lR['test_precision'].mean(),
                                                       lR['test_recall'].mean(),
                                                       lR['test_f1_score'].mean()]},

                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])

    # Add 'Best Score' column
    
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    return(models_scores_table)
models_evaluation(x, y, 3)

clf = AdaBoostClassifier(random_state = RAN_STATE)
x_tr = X_train
x_te = X_test
train_feature_list = [x_tr[0:10000],x_tr[0:20000],x_tr]
train_target_list = [y_train[0:10000], y_train[0:20000], y_train]
for a, b in zip(train_feature_list, train_target_list):
    clf.fit(a, b)
clf1.predict(X_test)


filename = 'Adaboost.pkl'
joblib.dump(clf1, filename)

