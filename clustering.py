
from libraries import *

'''
Este módulo tiene tres partes:
(1) Clase Clusters_RF con distintas funciones para el armado inicial de clusters usando K-Medoids y Random Forest
(2) Funciones de comparación entre clusters
(3) Función para entrenar mútliples modelos de regresión
'''


# Armado inicial de clusters usando K-Medoids y Random Forest ---------------------

class Clusters_RF:
    
    '''
    Podemos calcular distintas instancias de Clusters_RF
    - Desde distintos datos de partida
    - Obtiene K clusters de lemmas
    - Partiendo de la semilla rseed
    - Ordenados según los más predictores de que una película sea de inmigración
    '''
    
    
    def __init__(self, 
                  l2v, distance_matrix, filmids, tfidf, 
                  k, rseed):
           
        '''
        Cada instancia queda definida por:
        
        DATOS DE PARTIDA
        - distance_matrix
        - l2v
        - filmids
        - tfidf
        
        PARÁMETROS DE CLUSTERING
        - rseed
        - K número de clusters
        
        PARÁMETROS RANDOM FOREST
        - yvar
        '''
        
        self.l2v = l2v
        self.distance_matrix = distance_matrix
        self.filmids = filmids
        self.tfidf = tfidf
        
        self.k = k
        self.rseed = rseed
        
        self.yvar = "just_migra"
        self.rseedrf = 42
       

    def get_clusters(self):

        kmclusters = kmedoids.fasterpam(diss = self.distance_matrix,
                                        medoids = self.k,
                                        random_state = self.rseed,
                                        n_cpu = 4)


        self.kmclusters =  kmclusters
    
  
    
    def describe_clusters(self):
        
        
        # Crear un dataset con el nombre del cluster y los lemas pertenecientes al cluster
        self.l2v['cluster' + str(self.k) ] = self.kmclusters.labels

        # Create a dataset describing the clusters
        clusters = self.l2v.groupby('cluster' + str(self.k), as_index = False).agg({"lemma": ["nunique", "unique"]})
        clusters.columns = ["cluster", "n_lemmas", "lemmas"]
        
        self.clusters = clusters
        
       
        
    def get_silhouette(self):
        '''
        Obtener el silhouette score
        '''
        dims = [c for c in self.l2v.columns if "dim_" in c]
        
        silhouette = silhouette_score(self.l2v[dims], self.kmclusters.labels, metric='cosine')
        self.silhouette = silhouette
        
       
        
    def get_f2c(self):
        '''
        Obtener matrix F2C: valor de cada película en cada cluster vía valores TFIDF
        '''
        
        # Matriz que asigna 1 si el lema pertenece al cluster
        self.l2v["aux"] = 1
        sparse_matrix = sp.coo_matrix( (  self.l2v['aux'], 
                                        ( self.l2v["stoi"],  self.l2v['cluster' + str(self.k)])) )
        
        peli_cluster = self.tfidf.dot(sparse_matrix)

        peli_cluster = pd.DataFrame(peli_cluster.toarray()).reset_index()
        peli_cluster = peli_cluster.rename(columns = {"index" : "filmid"})
        peli_cluster = peli_cluster.merge(self.filmids,
                                          how = "left",
                                          on = "filmid")

        self.f2c = peli_cluster 
        
        

    def rf(self):


        '''
        RANDOM FOREST
        '''


        # Definimos nuestras variables
        X = self.f2c[np.arange(0, self.k)] # cluster vars are named from 0 to k
        y = self.f2c[self.yvar]


        ## Hacemos el split train-test
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, 
                                                            random_state = self.rseedrf)


        # Random Forest Classifier
        # a. Tuneamos los hiperparámetros por CV
        rfclass = RandomForestClassifier(random_state=self.rseedrf)
        parameters = {'n_estimators': [ 200],
                      'max_features': ['sqrt'],
                      'max_depth' : [5,7],
                      'criterion' :['entropy'],
                      'min_impurity_decrease': [0.001,0.01,0.1],
                      'random_state': [self.rseedrf]}
        grid_obj = GridSearchCV(rfclass, parameters, scoring='roc_auc',cv=5)
        grid_fit = grid_obj.fit(X_train, y_train)
        rfclass_best_params = grid_obj.best_params_

        # b. Ajustamos el modelo y predecimos
        rfclass = RandomForestClassifier(**rfclass_best_params)
        rfclass = rfclass.fit(X_train,y_train)

        ## c. Creamos un data frame con la importancia de cada variable
        feature_importances = pd.DataFrame(rfclass.feature_importances_)
        feature_importances = pd.concat([feature_importances,pd.DataFrame(X.columns)], axis=1)
        feature_importances.columns = ['Importance','cluster']
        feature_importances = feature_importances.sort_values('Importance',ascending=False).reset_index(drop=True)
        feature_importances['Cumulative'] = np.cumsum(feature_importances['Importance'])

        # d. Guardamos una medida de cuan buena fue la prediccion
        y_test_pred_rfclass = rfclass.predict(X_test)
        roc_auc = roc_auc_score(y_test,y_test_pred_rfclass)

        self.feature_importances = feature_importances
        self.roc_auc = roc_auc 

# Medidas de comparación entre clusters ----------------------------
        
def prop_intersection(c1, c2):
    
    '''
    Proporción de lemas compartidos, siendo ca1 y c2 listas de lemas
    - Si todos los lemas son iguales, unión = intersección, y la proporción será 1
    '''
    
    w1 = set(c1)
    w2 = set(c2)
    prop = len(w1.intersection(w2)) / len(w1.union(w2)) 
    
    return prop

def mean_cos_sim(c1, c2, w2v):
    
    '''
    Promedio de la similaridad coseno entre los lemas de dos clusters
    Inputs:
    - c1, c2: listas de lemas en cada cluster
    - w2v: embedding para los lemas en los clusters
    
    '''
    
    c1_vectors =  w2v.loc[w2v.lemma.isin(c1),dims]
    c2_vectors =  w2v.loc[w2v.lemma.isin(c2),dims]
    cosine_sim_matrix = cosine_similarity(c1_vectors, c2_vectors)
   
    cos_sim = np.mean(cosine_sim_matrix)
  
    return cos_sim


# Modelos de regresión -----------------------------------

def regression_methods(X, y):
    '''
    Función que entrena modelos de regresión
    - Input: matriz de features X, variable dependiente y
    - Output: predicciones en train y test, hiperparámetros crosvalidados
    '''
    
    
    ## Creamos una lista donde vamos a guardar todos los hiperparámetros ajustados
    hyperparameters = {}
    predicted_train = pd.DataFrame()
    predicted_test  = pd.DataFrame()
    
    ## Hacemos el split train-test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 42)

    predicted_train["y_train"] = y_train
    predicted_test["y_test"]  = y_test

    # 0. Dummy regressor
    dummy_regr1 = DummyRegressor(strategy="mean")
    dummy_regr1 = dummy_regr1.fit(X_train, y_train)
    predicted_train["mean"] = dummy_regr1.predict(X_train)
    predicted_test["mean"]  = dummy_regr1.predict(X_test)
    
    dummy_regr2 = DummyRegressor(strategy="median")
    dummy_regr2 = dummy_regr2.fit(X_train, y_train)
    predicted_train["median"] = dummy_regr2.predict(X_train)
    predicted_test["median"]  = dummy_regr2.predict(X_test)
    
        
    # 1. Linear Regresion 
    lr = LinearRegression()
    lr = lr.fit(X_train, y_train)
    predicted_train["lr"] = lr.predict(X_train)
    predicted_test["lr"] = lr.predict(X_test)
    
                                 
    # 2. Lasso
    # a. Tuneamos los hiperparámetros por CV
    lasso = Lasso()
    parameters = {'alpha': [0.0001, 0.001 , 0.01, 0.1, 0.5, 1, 2, 50], 
                 'selection': ['random'],
                  'max_iter': [30000],
                  'random_state': [9]
                 }
    
    
    grid_obj = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_fit = grid_obj.fit(X_train,y_train)
    lasso_best_params = grid_obj.best_params_
    hyperparameters['Lasso'] = lasso_best_params
    # b. Ajustamos el modelo y predecimos
    lasso = Lasso(**lasso_best_params)
    lasso = lasso.fit(X_train, y_train)
    predicted_train["lasso"]  = lasso.predict(X_train)
    predicted_test["lasso"] = lasso.predict(X_test)
    
    
    
    # 3. Ridge
      # a. Tuneamos los hiperparámetros por CV
    ridge = Ridge()
    parameters = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 20, 50], 
                  'max_iter': [30000],
                  'random_state': [9]
                 }
    
    
    grid_obj = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_fit = grid_obj.fit(X_train,y_train)
    ridge_best_params = grid_obj.best_params_
    hyperparameters['Ridge'] = ridge_best_params
    # b. Ajustamos el modelo y predecimos
    ridge = Ridge(**ridge_best_params)
    ridge = ridge.fit(X_train, y_train)
    predicted_train["ridge"] = ridge.predict(X_train)
    predicted_test ["ridge"] = ridge.predict(X_test)

    # 4. ElasticNet
    # a. Tuneamos los hiperparámetros por CV
    en = ElasticNet()
    parameters = {'alpha': [0.001, 0.01, 0.1, 2], 
              'l1_ratio': [0.2, 0.5, 0.8], 
                  'selection': ['random'],
                  'max_iter': [10000],
                  'random_state': [9]
                 }
    
    
    grid_obj = GridSearchCV(en, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_fit = grid_obj.fit(X_train,y_train)
    en_best_params = grid_obj.best_params_
    hyperparameters['ElasticNet'] = en_best_params
    # b. Ajustamos el modelo y predecimos
    en = ElasticNet(**en_best_params)
    en = en.fit(X_train, y_train)
    predicted_train["en"] = en.predict(X_train)
    predicted_test ["en"] =en. predict(X_test)

    # 5 Random Forest Regression
    # a. Tuneamos los hiperparámetros por CV
    rf = RandomForestRegressor(random_state=42)
    parameters = {'n_estimators': [100, 200],
                  'max_features': ['sqrt'],
                  'max_depth' : [5,9],
                  'criterion' :["squared_error"],
                  'min_impurity_decrease': [0]}
    grid_obj = GridSearchCV(rf, parameters, scoring='neg_mean_squared_error', cv=5)
    grid_fit = grid_obj.fit(X_train, y_train)
    rf_best_params = grid_obj.best_params_
    hyperparameters['RandomForest'] = rf_best_params
    # b. Ajustamos el modelo y predecimos
    rf = RandomForestRegressor(**rf_best_params, random_state=42)
    rf = rf.fit(X_train,y_train)
    predicted_train["rf"] = rf.predict(X_train)
    predicted_test ["rf"]  = rf.predict(X_test)
 
         
    # GUARDAMOS PREDICCIONES
    return predicted_train, predicted_test, hyperparameters