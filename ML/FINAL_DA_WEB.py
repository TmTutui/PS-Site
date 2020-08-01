def ml_model(inputer):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    # %%
    inputFromWeb = pd.DataFrame()


    # %%
    holder = pd.DataFrame()
    holder['tpv medio'] = [inputer[0]]
    holder['segmento_ord'] = inputer[1]
    holder['mcc_ord'] = inputer[2]
    holder['bairro_ord'] = inputer[3]
    holder['cidade_ord'] = inputer[4]
    holder['regiao_ord'] = inputer[5]
    holder['uf_ord'] = inputer[6]
    holder['subcanal_de_vendas_ord'] = inputer[7]
    inputFromWeb = inputFromWeb.append(holder)



    # %%
    credito = pd.read_csv("ML/base_credito_treino_LIMPA.csv", index_col = 0)
    tpv = pd.read_csv("ML/base_tpv2_treino_LIMPA.csv", index_col = 0)
    rav = pd.read_csv("ML/base_rav2_treino_LIMPA.csv", index_col = 0)
    cadastro_treino = pd.read_csv("ML/base_cadastro_treino_LIMPA.csv", index_col = 0)
    cadastro_treino_dummies = pd.read_csv("ML/base_cadastro_treino_DUMMIES_LIMPA.csv", index_col= 0)
    cadastro_treino_ordinal = pd.read_csv("ML/base_ordinal_LIMPA.csv", index_col= 0)
    quanti_do_sid = pd.read_csv("ML/tabela_quanti_do_sid.csv", index_col= 0)


    # %%
    credito = credito.sort_values(by = ['client_id'])
    cadastro_treino = cadastro_treino.sort_values(by = ['client_id'])
    cadastro_treino_dummies = cadastro_treino_dummies.sort_values(by = ['client_id'])
    cadastro_treino_ordinal = cadastro_treino_ordinal.sort_values(by = ['client_id'])
    rav = rav.sort_values(by = ['client_id'])
    tpv = tpv.sort_values(by = ['client_id'])
    quanti_do_sid = quanti_do_sid.sort_values(by = ['client_id'])


    # %%
    """ seleciona todas as colunas com algo no nome """
    selected_cols = [col for col in cadastro_treino_ordinal.columns if ('ord') in col]


    # %%
    quanti_do_sid[selected_cols] = cadastro_treino_ordinal[selected_cols]
    quanti_do_sid['limite_binado'] = credito['limite_binado']


    # %%
    """ dependendo do dataset q eu vou usar eu mudo qual linha comentada ta comentada """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(quanti_do_sid[['tpv medio','segmento_ord', 'mcc_ord', 'bairro_ord', 'cidade_ord', 'regiao_ord', 'uf_ord', 'subcanal_de_vendas_ord']], credito[['default_60d', 'limite', 'default_mar']], test_size = 0.2, random_state=123)


    # %%
    from sklearn.preprocessing import StandardScaler

    X_train_for_scaling = X_train
    X_test_for_scaling = X_test

    scaler = StandardScaler().fit(X_train_for_scaling)
    scaler.transform(X_train_for_scaling)
    scaler.transform(X_test_for_scaling)
    """ variáveis foram clonadas para não afetar os outros modelos """

    # %% [markdown]
    # ## Ensemble:

    # %%
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # variáveis para plugar os dados e labels
    train_data = X_train_for_scaling
    train_labels = y_train['default_60d']
    test_data = X_test_for_scaling 
    test_labels = y_test['default_60d']


    # probabilidade que divide os inadimplentes / n inadimplentes
    THRESHOLD = 0.36

    # inicializando os modelos e salvando as probabilidades de cada um
    logreg = LogisticRegression().fit(train_data, train_labels)
    logreg_pred_threshold = np.where(logreg.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    logreg_pred_proba = logreg.predict_proba(inputFromWeb)

    xgb = XGBClassifier().fit(train_data, train_labels)
    xgb_pred_threshold = np.where(xgb.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    xgb_pred_proba = xgb.predict_proba(inputFromWeb)

    kneighbors = KNeighborsClassifier(n_neighbors = 5).fit(train_data, train_labels)
    kneighbors_pred_threshold = np.where(kneighbors.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    kneighbors_pred_proba = kneighbors.predict_proba(inputFromWeb)

    average_probability_prediction_default_60d = []


    # %%
    # obs: o primeiro valor da dupla de probabilidades é o que analisamos o THRESHOLD: se estiver abaixo é inadimplente
    average_probability_prediction_default_60d = []
    for i in range(len(logreg_pred_threshold)):
        average_probability_prediction_default_60d.append((logreg_pred_proba[i] + xgb_pred_proba[i] + kneighbors_pred_proba[i])/3)

    average_probability_prediction_default_60d = np.array(average_probability_prediction_default_60d)
    ENSEMBLE_THRESHOLD = 0.33
    average_probability_threshold_default_60d = np.where(average_probability_prediction_default_60d[:,1] > ENSEMBLE_THRESHOLD, 1, 0)
    prob_default_60d = pd.DataFrame(average_probability_prediction_default_60d)[0]


    # %%

    # variáveis para plugar os dados e labels
    train_data = X_train_for_scaling
    train_labels = y_train['default_mar']
    test_data = X_test_for_scaling 
    test_labels = y_test['default_mar']


    # probabilidade que divide os inadimplentes / n inadimplentes
    THRESHOLD = 0.36

    # inicializando os modelos e salvando as probabilidades de cada um
    logreg = LogisticRegression().fit(train_data, train_labels)
    logreg_pred_threshold = np.where(logreg.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    logreg_pred_proba = logreg.predict_proba(inputFromWeb)

    xgb = XGBClassifier().fit(train_data, train_labels)
    xgb_pred_threshold = np.where(xgb.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    xgb_pred_proba = xgb.predict_proba(inputFromWeb)

    kneighbors = KNeighborsClassifier(n_neighbors = 5).fit(train_data, train_labels)
    kneighbors_pred_threshold = np.where(kneighbors.predict_proba(inputFromWeb)[:,1] > THRESHOLD, 1, 0)
    kneighbors_pred_proba = kneighbors.predict_proba(inputFromWeb)

    average_probability_prediction_default_mar = []


    # %%
    # obs: o primeiro valor da dupla de probabilidades é o que analisamos o THRESHOLD: se estiver abaixo é inadimplente
    average_probability_prediction_default_mar = []
    for i in range(len(logreg_pred_threshold)):
        average_probability_prediction_default_mar.append((logreg_pred_proba[i] + xgb_pred_proba[i] + kneighbors_pred_proba[i])/3)

    average_probability_prediction_default_mar = np.array(average_probability_prediction_default_mar)
    ENSEMBLE_THRESHOLD = 0.33
    average_probability_threshold_default_mar = np.where(average_probability_prediction_default_mar[:,1] > ENSEMBLE_THRESHOLD, 1, 0)


    # %%
    inputFromWeb['prob_default_60d'] = prob_default_60d

    # %% [markdown]
    # # Usando XGB para estimar limite, juros e afins a partir da inadimplencia
    # %% [markdown]
    # ## Limite

    # %%
    """ dependendo do dataset q eu vou usar eu mudo qual linha comentada ta comentada """

    X = pd.DataFrame()
    X[['tpv medio','segmento_ord', 'mcc_ord', 'bairro_ord', 'cidade_ord', 'regiao_ord', 'uf_ord', 'subcanal_de_vendas_ord']] = quanti_do_sid[['tpv medio','segmento_ord', 'mcc_ord', 'bairro_ord', 'cidade_ord', 'regiao_ord', 'uf_ord', 'subcanal_de_vendas_ord']]
    X['prob_default_60d'] = pd.DataFrame(average_probability_prediction_default_60d)[0]


    X_train, X_test, y_train, y_test = train_test_split(X, credito[['limite', 'juros_am']], test_size = 0.2, random_state=123)


    # %%

    X_train_for_scaling = X_train
    X_test_for_scaling = X_test

    scaler = StandardScaler().fit(X_train_for_scaling)
    scaler.transform(X_train_for_scaling)
    scaler.transform(X_test_for_scaling)
    """ variáveis foram clonadas para não afetar os outros modelos """


    # %%
    from xgboost import XGBRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsRegressor


    # variáveis para plugar os dados e labels
    train_data = X_train_for_scaling
    train_labels = y_train['limite']
    test_data = X_test_for_scaling 
    test_labels = y_test['limite']

    model = XGBRegressor().fit(train_data, train_labels)

    estimativa_limite = model.predict(inputFromWeb)


    # %%
    """ dependendo do dataset q eu vou usar eu mudo qual linha comentada ta comentada """

    X = pd.DataFrame()
    X[['tpv medio','segmento_ord', 'mcc_ord', 'bairro_ord', 'cidade_ord', 'regiao_ord', 'uf_ord', 'subcanal_de_vendas_ord']] = quanti_do_sid[['tpv medio','segmento_ord', 'mcc_ord', 'bairro_ord', 'cidade_ord', 'regiao_ord', 'uf_ord', 'subcanal_de_vendas_ord']]
    X['prob_default_60d'] = pd.DataFrame(average_probability_prediction_default_60d)[0]
    X['prob_default_mar'] = pd.DataFrame(average_probability_prediction_default_mar)[0]


    # %%
    X_train_for_scaling = X_train
    X_test_for_scaling = X_test

    scaler = StandardScaler().fit(X_train_for_scaling)
    scaler.transform(X_train_for_scaling)
    scaler.transform(X_test_for_scaling)
    """ variáveis foram clonadas para não afetar os outros modelos """


    # variáveis para plugar os dados e labels
    train_data = X_train_for_scaling
    train_labels = y_train['juros_am']
    test_data = X_test_for_scaling 
    test_labels = y_test['juros_am']

    model = XGBRegressor().fit(train_data, train_labels)


    estimativa_juros = model.predict(inputFromWeb)

    # %% [markdown]
    # ## Finalizando

    # %%

    inputFromWeb['estimativa_limite'] = estimativa_limite

    inputFromWeb['estimativa_juros_parcelados'] = estimativa_juros

    inputFromWeb['estimativa_juros_rotativo'] = inputFromWeb['estimativa_juros_parcelados'].apply(lambda x: 2.5*x)

    inputFromWeb.to_csv('OFERTA-ANALISADO.csv')

    return estimativa_limite[0], estimativa_juros[0], inputFromWeb['prob_default_60d'][0]



