import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
selected = option_menu(
    "Menu",
    ["Acceuil", "EDA", "Prédiction"],
    icons=["house", "graph-up", "robot"],
    menu_icon="menu-app",
    default_index=0
    )
def Acceuil():
    st.markdown("""
                ## 🎉 Bienvenue dans l’application de segmentation clients !

                Cette application a été développée pour explorer, analyser et segmenter les clients d’une grande boutique grâce aux techniques de Data Science, de Machine Learning et de Traitement Automatique du Langage (NLP).

                ---

                ### 📌 À propos du projet
                - **Auteur** : Ange DAHOU, étudiant en Physique-Chimie et aspirant Data Scientist 🧠

                - **Objectif** : Identifier des segments de clients et comprendre leurs comportements d’achat afin d’optimiser les décisions marketing 📊

                - **Technologies utilisées** : Python, Pandas, Scikit-learn, Matplotlib, Plotly Express, Streamlit 🚀

                ---

                ### 📂 À propos du dataset

                - **Nom du dataset** : `data.csv`
                - **Nombre d’observations** : `541909`
                - Informations transactionnelles (quantité, prix, date d’achat…)
                - Descriptions produits (texte en anglais, analysé via NLP)

                > 🔎 Ce projet vous propose une navigation simple :  
                > - Analyse exploratoire (EDA)  
                > - Prédiction des clusters

                ---

                ### 📌 Contexte et problématique

                > Dans un marché fortement concurrentiel, les entreprises doivent non seulement attirer de nouveaux clients, mais aussi fidéliser ceux déjà existants. Comprendre les profils de clients et les produits phares par segment devient donc crucial.

                Ce projet s’appuie sur :
                - L’analyse exploratoire (EDA) pour identifier les tendances,
                - Le clustering pour regrouper les clients selon leurs comportements,
                - Le NLP pour analyser les descriptions produits et détecter les articles les plus représentatifs de chaque cluster.

                ---

                ### ❓ Problématique

                > Quels sont les différents profils de clients de la boutique, et quels produits phares caractérisent chaque segment ?

                Répondre à cette question permettra :
                - de mieux cibler les campagnes marketing,
                - de mettre en avant les bons produits auprès des bons segments,
                - et d’améliorer la rentabilité en réduisant les actions inefficaces.

                ---
                ### 📌 Méthodologie  
                L’approche suivie pour ce projet repose sur plusieurs étapes complémentaires :  

                1. **Analyse exploratoire des données (EDA) 🔍**  
                - Étude des tendances d’achat par période, par continent et par produit.  
                - Détection des comportements récurrents afin de mieux comprendre les clients.  

                2. **Segmentation par Clustering 📊**
                - Utilisation de l’algorithme *KMeans* pour regrouper les clients selon leurs comportements d’achat.  
                - Détermination du nombre optimal de clusters à l’aide de métriques (méthode du coude, silhouette score).  

                3. **Analyse textuelle (NLP) 📝**  
                - Traitement des descriptions produits (en anglais).  
                - Identification des *produits phares représentatifs de chaque cluster*, afin de relier les profils clients aux articles achetés.  

                4. **Prédiction et Modélisation 🤖** 
                - Mise en place de modèles supervisés avec *Scikit-learn*.  
                - Objectif : prédire à quel cluster appartient un nouveau client selon ses caractéristiques.  
                ---
                ### 📌 À savoir
                Le dataset original contenait plus de 500.000 observations. Cependant, l’algorithme KMeans nécessite une quantité de mémoire très élevée (plus d'1 To de RAM) pour traiter l’ensemble du jeu de données.
                Afin de rendre l’analyse possible, un échantillonnage stratifié par continent a été réalisé, réduisant ainsi le dataset à 10 000 observations représentatives.

                ---

                ### 👉 Bon passage sur l’application et bonne exploration ! 🚀
                """)

def EDA_page():
    st.markdown("""
                ## 🔍 Analyse exploratoire (EDA)

                Bienvenue dans la section EDA !  
                Ici, vous pouvez explorer les données brutes pour en découvrir la structure, les tendances, les anomalies, et les insights essentiels.

                ---

                ### 🧰 Outils utilisés :
                - Statistiques descriptives
                - Visualisations interactives (matplotlib/seaborn)
                - Analyse des distributions, corrélations et valeurs manquantes

                > 🎯 L’objectif est de mieux **comprendre** les données avant d’entraîner un modèle prédictif.
                ---
                """)
    @st.cache_data
    def load_data():
        df = pd.read_csv("data_clusters.csv")

        return df

    df = load_data()

    df1 = df.copy()
    dow_map = {
            0: 'Lundi',
            1: 'Mardi',
            2: 'Mercredi',
            3: 'Jeudi',
            4: 'Vendredi',
            5: 'Samedi',
            6: 'Dimanche'}

    df1['dow'] = df1['dow'].replace(dow_map)
    month_map = {
        0: 'Janvier',
        1: 'Février',
        2: 'Mars',
        3: 'Avril',
        4: 'Mai',
        5: 'Juin',
        6: 'Juillet',
        7: 'Août',
        8: 'Septembre',
        9: 'Octobre',
        10: 'Novembre',
        11: 'Décembre',
        12: 'Janvier'}

    df1['month'] = df1['month'].replace(month_map)
    #with st.expander('Affichez le dataset'):
        #st.write(df1)
    boutton = st.sidebar.radio(
            "Option",
            ["Analyse visuelle", "Rapport"]
        )
    if boutton=="Analyse visuelle":
        EDA = st.sidebar.checkbox("Analyse de la Class", value= True)
        temps = st.sidebar.checkbox("Analyse Temporelle", value= True)
        region = st.sidebar.checkbox("Analyse Géographique", value= True)
        st.header("Analyse Exploratoire des Données (EDA)")
        if EDA:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Repartition des observations par cluster')
                cluster_counts = df['clusters'].value_counts(normalize=True).reset_index()
                cluster_counts.columns = ['Cluster', 'Proportion']
                fig = px.pie(
                    cluster_counts,
                    values='Proportion',
                    names='Cluster',
                    title="Répartition des observations par cluster ",
                    hole=0.3,  
                    color_discrete_sequence=px.colors.qualitative.Set3
                )

                fig.update_traces(
                    textinfo="percent+label",
                    textposition="inside"
                )

                fig.update_layout(
                    showlegend=True,
                    template="plotly_dark",
                    height=500,width=900
                )

                st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.subheader('Dépense moyenne des clients par clusters')

                    cts = df.groupby('clusters')['PT'].mean().reset_index()

                    fig = px.pie(cts, 
                                names='clusters', 
                                values='PT', 
                                title="Dépense moyenne par clusters",
                                color='clusters',      
                                template="plotly_dark",
                                hole=0.3)           

                    fig.update_traces(textposition='inside', textinfo='label+value')
                    fig.update_layout(height=500,width=900)

                    st.plotly_chart(fig, use_container_width=True)

        if temps:    
            col3, col4 = st.columns(2)
            with col3:
                st.subheader('Evolution des dépenses totales des clients par mois')
                cts = df.groupby('month')['PT'].sum().reset_index()

                fig = px.line(cts,
                            x="month",
                            y="PT",
                            title="Depense total  par mois",
                            markers=True)

                fig.update_layout(xaxis_title="mois",
                                yaxis_title="Depense totale",
                                template="plotly_dark",  
                                height=500,width=900)
                
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                st.subheader('Evolution des dépenses totales des clients par jours')
                dow_users = df.groupby('dow')['PT'].sum().reset_index()

                fig = px.line(dow_users,
                            x="dow",
                            y="PT",
                            title="Dépenses totales des Clients par Jours",
                            markers=True)

                fig.update_layout(xaxis_title="Jours de la semaine",
                                yaxis_title="Depenses",
                                template="plotly_dark",
                                height=500,width=900)
                
                st.plotly_chart(fig, use_container_width=True)

            st.subheader('Evolution des dépenses totales des Clients par Heure')
            hour_counts = df.groupby('heure')['PT'].sum().reset_index()

            fig = px.line(hour_counts,
                        x="heure",
                        y="PT",
                        title="Dépenses totales des Clients par Heure",
                        markers=True)

            fig.update_layout(xaxis_title="Heure de la journée",
                            yaxis_title="Nombres de clients",
                            template="plotly_dark",  
                            height=500,width=700)
            st.plotly_chart(fig, use_container_width=True)
        if region:
            col7, col8 = st.columns(2)
            with col7:
            
                st.subheader('Répartition des clusters par continent')
                ct = pd.crosstab(df['Continent'], df['clusters'], normalize='index') * 100
                continent_order = df['Continent'].value_counts().index
                ct = ct.loc[continent_order]

                ct_long = ct.reset_index().melt(id_vars='Continent', var_name='Cluster', value_name='Pourcentage')

                fig = px.bar(
                    ct_long,
                    x="Pourcentage",
                    y="Continent",
                    color="Cluster",
                    orientation="h",
                    title="Répartition des clusters par continent (%)",
                    text="Pourcentage",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig.update_traces(texttemplate='%{text:.1f}%', textposition="inside")
                fig.update_layout(
                    barmode="stack",  
                    xaxis_title="Pourcentage (%)",
                    yaxis_title="Continent",
                    template="plotly_white",
                    height=500,width=900
                )

                st.plotly_chart(fig, use_container_width=True)
                


            with col8:

                st.subheader('Dépenses totales moyenne par continent')
                cts = df.groupby('Continent')['PT'].mean().reset_index()

                fig = px.bar(
                    cts,
                    x="Continent",
                    y="PT",
                    text="PT",
                    orientation="v",
                    title="Dépenses totales moyenne par continent",
                    color="Continent",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig.update_traces(texttemplate='%{text:.1f}', textposition="outside")
                fig.update_layout(
                    xaxis_title="Continent",
                    yaxis_title="Dépense totale moyenne",
                    template="plotly_dark",  
                    height=500,
                    width=900,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

            st.subheader('Répartition des clusters par Mois')
            df['month'] = df['month'].astype(int)
            ct = pd.crosstab(df['month'], df['clusters'], normalize='index') * 100
            ct = ct.sort_index()  

            ct_long = ct.reset_index().melt(id_vars='month', var_name='Cluster', value_name='Pourcentage')

            fig = px.bar(
                ct_long,
                x="Pourcentage",
                y="month",
                color="Cluster",
                orientation="h",
                title="Répartition des clusters par mois (%)",
                text="Pourcentage",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig.update_traces(texttemplate='%{text:.1f}%', textposition="inside")
            fig.update_layout(
                barmode="stack",   
                xaxis_title="Pourcentage (%)",
                yaxis_title="Mois",
                template="plotly_white",
                height=600,width=900,
                yaxis=dict(dtick=1)  
            )
            st.plotly_chart(fig, use_container_width=True)
        
        
            
    elif boutton == "Rapport":
        rap_EDA = st.sidebar.checkbox("Rapport de l'EDA", value=True)
        rap_RFM = st.sidebar.checkbox("Rapport du RFM", value=True)
        if rap_EDA:
            st.markdown("""
            ## Rapport d’Analyse Client – Clustering & Comportement d’Achat

            
            Cette analyse se concentre sur le comportement d’achat des clients selon leur cluster. L’objectif est d’identifier les profils clés, comprendre les habitudes d’achat, la saisonnalité, et les préférences produits. Ces informations permettront d’optimiser le ciblage marketing, la gestion des stocks et la planification des campagnes.

            ### Répartition des Clusters
            Les clients se répartissent de manière hétérogène entre les 4 clusters : 
                        
            •   Cluster 0 représente 18,1%, 
                        
            •   Cluster 1 représente 28,2%, 
                        
            •   Cluster 2 représente 32,8%, et 
                        
            •   Cluster 3 représente 20,8%.
        
            #### Insight 
                        
                Le Cluster 2 concentre la plus grande part de clients, tandis que le Cluster 0 reste le plus restreint. 
                Cette répartition montre que chaque cluster a un poids significatif et doit être intégré dans la stratégie globale.


            

            ### Répartition Géographique des Clusters
                Les clients proviennent de 5 zones : Asie, Océanie, Europe, Amérique et unknown. 
                Tous les continents sont représentés dans les 4 clusters, à l’exception de l’Amérique, où aucun client n’appartient au Cluster 3.
            
            #### Insight 
                        
                Cette répartition géographique équilibrée confirme la diversité du portefeuille client, mais l’absence du Cluster 3 en Amérique peut signaler une opportunité de pénétration du marché ou un décalage d’offre produit.

            

            ### Dépenses par Cluster
            L’analyse révèle que les clients du Cluster 0 et du Cluster 2 dépensent en moyenne 14€, ceux du Cluster 1 légèrement plus (16€), alors que le Cluster 3 se distingue nettement avec 36€ de dépense moyenne.
                        
            #### Insight
                        
                Malgré sa taille plus réduite, le Cluster 3 a un pouvoir d’achat largement supérieur, ce qui en fait une cible prioritaire pour les offres premium ou à forte valeur ajoutée.

            
            ### Saisonnalité et Mois d’Achat
            •   Cluster 0 : Achète sur les 5 derniers mois, contribuant de 19% à 42% des ventes mensuelles.
                        
            •	Cluster 1 : Achète exclusivement sur les 6 premiers mois, représentant entre 52% et 89% des achats mensuels.
                        
            •	Cluster 2 : Actif uniquement sur les 6 derniers mois, concentrant entre 47% et 58% des ventes mensuelles.
                        
            •	Cluster 3 : Présent toute l’année, avec une contribution allant de 10% à 47%.

            #### Insight 
                        
                Le Cluster 3 est le plus régulier, soutenant les ventes toute l’année, alors que les autres clusters sont fortement liés à des périodes spécifiques. 
                Cela suggère de développer des campagnes différenciées par saison : offres de début d’année pour le Cluster 1, promotions de fin d’année pour les Clusters 0 et 2.

            
            ### Plages Horaires
            La boutique est ouverte de 6h à 20h.
                        
            •	Entre 6h et 9h, les ventes augmentent fortement, passant de 5€ à 25 000€.
                        
            •	De 9h à 15h, le niveau reste stable, oscillant entre 20 000€ et 25 000€, avec des pics à 9h et 15h.
                        
            •	Entre 15h et 20h, les ventes décroissent progressivement jusqu’à retomber à 5€.

            #### Insight 
                        
                Les créneaux 9h et 15h sont les moments clés de la journée. 
                Une intensification des actions commerciales (promotions, mise en avant de produits) sur ces heures pourrait maximiser les ventes.


            ### Analyse NLP – Description des Produits
            •	Cluster 0 : Les mots les plus fréquents sont set, bag, christmas, vintage, red, heart, retrospot, design, pink, cake.
                        
            → Les clients de ce cluster sont particulièrement attirés par les articles festifs et saisonniers (Christmas, heart, red, vintage), associés aux cadeaux de fin d’année.
            
            •	Cluster 1 : Les mots les plus fréquents sont bag, retrospot, red, set, pink, heart, design, blue, cake, white.
                        
            → Ici, l’accent est mis sur les accessoires colorés et de design (bag, retrospot, pink, blue, white). Ces clients semblent sensibles au style et à la variété des articles plutôt qu’à la saisonnalité.
                        
            •	Cluster 2 : Les mots les plus fréquents sont set, bag, christmas, heart, red, vintage, retrospot, design, box, pink.
                        
            → Ce cluster montre un panier d’achat mixte, combinant des produits festifs (Christmas, heart, vintage) et des articles pratiques (box, set, bag), suggérant des achats réguliers mais aussi liés aux événements.
                        
            •	Cluster 3 : Les mots les plus fréquents sont bag, set, red, heart, retrospot, pink, vintage, design, jumbo, box.
                        
            → Ce cluster met en avant des articles de grande taille ou utilitaires (jumbo, box), en plus des classiques (bag, set, red). Cela indique un intérêt pour des articles plus fonctionnels et variés, adaptés à une consommation diversifiée.

            #### Insights
                        
                •	Clusters 0 & 2 → orientés festifs et cadeaux saisonniers, en lien avec Noël et les événements de fin d’année.
                •	Cluster 1 → davantage centré sur les articles de design et mode, sensibles aux couleurs et aux tendances.
                •	Cluster 3 → profil plus utilitaire et varié, avec un attrait pour des produits de grande taille ou pratiques.
                        

            ### Conclusion & Recommandations
                •	Le Cluster 3 se démarque par son dépense moyenne élevée (36€) et sa présence continue toute l’année, ce qui en fait un segment stratégique.
                        
                •	Les heures de pointe (9h & 15h) et les jours forts (mardi & jeudi) doivent être renforcés par des promotions ciblées.
                        
                •	La saisonnalité des clusters impose une segmentation des campagnes : début d’année pour Cluster 1, fin d’année pour Clusters 0 et 2, toute l’année pour Cluster 3.
                        
                •	L’absence de ventes le samedi est une limite opérationnelle à reconsidérer.
                        
                •	Une stratégie spécifique pourrait être développée en Amérique pour introduire le Cluster 3.


            ## Opportunités Marketing
                •	Mettre en avant des promotions ciblées clusters en fonction des besoins des clients.
                •	Adapter les stocks selon la saisonnalité et les clusters.
                •	Réévaluer la gestion des horaires, notamment l’ouverture le samedi.

            ---
            Cette analyse met en lumière des informations stratégiques pour mieux comprendre le comportement d’achat des clients et optimiser les performances commerciales. En exploitant ces résultats, la boutique pourra affiner son ciblage marketing, adapter ses stocks en fonction de la saisonnalité, et personnaliser ses offres selon les préférences des différents clusters. Cette approche permettra non seulement d’augmenter le chiffre d’affaires, mais aussi d’améliorer l’expérience client et de réduire les pertes liées à une gestion inefficace des campagnes.
            
            """)

        if rap_RFM:
            st.header("Analyse du RFM (Recency, Frequency, Monetary)")
            st.markdown("""
            L'analyse RFM est une technique de segmentation des clients basée sur trois dimensions clés :
            
            - **Récence (Recency)** : Depuis combien de temps le client a-t-il effectué son dernier achat ?
            - **Fréquence (Frequency)** : À quelle fréquence le client effectue-t-il des achats sur une période donnée ?
            - **Monétaire (Monetary)** : Quel est le montant total dépensé par le client sur une période donnée ?
            
            En combinant ces trois dimensions, les entreprises peuvent identifier et segmenter leurs clients en fonction de leur comportement d'achat. Cela permet de cibler efficacement les campagnes marketing, d'améliorer la fidélisation des clients et d'optimiser les stratégies de vente.
            """)
            @st.cache_data
            def load_rfm():
                df_rfm = pd.read_csv("RFM.csv")
                return df_rfm
            
            df_rfm = load_rfm()
            with st.expander(f"Affichez le résultat de l'analyse RFM"):
                st.write(df_rfm)

            st.markdown("""
            | *Cluster* | *Profil Client* | *Stratégie Marketing* |
            |-------------|------------------|--------------------------|
            | *0* | Clients *récents* (Recency faible), *très actifs* (Frequency élevée), mais avec un *panier moyen modéré. Ce sont des acheteurs réguliers et fidèles. | Fidéliser avec un **programme de fidélité** (réductions cumulées, points, cadeaux), et proposer des ventes croisées (cross-selling) pour augmenter le panier moyen. |
            | *1* | Clients *peu récents* (longtemps inactifs), avec une *fréquence faible, mais un **panier moyen correct**. Ce sont des clients à risque de **churn**. | Relancer avec des **campagnes de réactivation** (emails personnalisés, offres exclusives, codes promo). Mettre en avant la *nouveauté* et les produits tendances pour susciter leur retour. |
            | *2* | Clients *récents*, avec une **fréquence moyenne** et un panier légèrement supérieur à la moyenne. Ils sont actifs mais pas encore pleinement fidèles. | Encourager la **récurrence d’achat** avec des *ventes flash* ou des *abonnements*. Créer une expérience personnalisée (recommandations basées sur achats passés). |
            | *3* | Clients *intermédiaires en récence** (ni très récents ni trop anciens), avec une *faible fréquence*, mais un **panier très élevé**. Ce sont des **gros dépensiers occasionnels**. | Mettre en place une **stratégie premium/VIP** (accès anticipé aux nouveautés, service personnalisé). Offrir des avantages exclusifs pour les inciter à acheter plus souvent. |
            """)


import joblib
def prediction():
    st.markdown("""
                ## 🤖 Prédiction automatique

                Dans cette section, vous pouvez tester le **modèle prédictif** entraîné sur notre dataset.

                ---

                ### ✨ Fonctionnalités :
                - Saisissez les caractéristiques d’un individu ou d’un événement
                - Obtenez une prédiction instantanée
                - Interprétation des résultats fournie

                > 📢 Le modèle a été sélectionné pour sa **précision** et sa **robustesse**.
                ---
                ### 📝 Renseigez les information svp
                """)
    @st.cache_resource
    def load_model():
        model = joblib.load("class_model.pkl")  
        return model

    model = load_model()



    # Entrer utilisateur 
    col1, col2, col3 = st.columns(3)

    col4, col5, col6 = st.columns(3)

    with col1:
        mois = st.selectbox(
            label=("Quelle est le mois d'achat de votre client"),
            options=['Janvier', 'Février', 'Mars', 'Avril', 'Mai',
                        'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
            )
        mois_map = {'Janvier' :0, 'Février':1, 'Mars':2, 'Avril':3, 'Mai':4,
                        'Juin':5, 'Juillet':6, 'Août':7, 'Septembre':8, 'Octobre':9, 'Novembre':10, 'Décembre':11}
        mois_num = mois_map[mois]

    with col2:
        dow = st.selectbox(
            label="Quelle est le mois d'achat de votre client", 
            options=['Lundi', 'Mardi', 
                    'Mercredi', 'Jeudi', 
                    'Vendredi', 'Samedi', 'Dimanche']
                    )
        
        dow_mapping = {'Lundi':0, 'Mardi':1, 
                    'Mercredi':2, 'Jeudi':3, 
                    'Vendredi':4, 'Samedi':5, 'Dimanche':6}
        dow_num = dow_mapping[dow]

    with col3:
        PU = st.number_input(label=('Quelle est le Prix Unitaire du produit acheté'), min_value=1, max_value=38970, value = 5)

    with col4:
        Quantite = st.number_input(label=('Quelle est la Quantité du produit acheté'), min_value=1, max_value=80995, value = 10)

    with col5:
        heure = st.number_input(label=("Quelle est l'heure d'achat de votre client"), min_value=6, max_value=20, value = 13)

    with col6:
        Continent = st.selectbox(label="Quelle est le continent de votre client", options=['Europe', 'Unknown', 'Oceania', 'Asia', 'America'])

        Continent_mapping = {'Europe':0, 'Unknown':1, 'Oceania':2, 'Asia':3, 'America':4}

        Continent_num = Continent_mapping[Continent]
    PT = Quantite*PU
    def inference(Quantite, PU, heure, mois_num, dow_num, PT, Continent_num):
                  
        new_data = pd.DataFrame([[Quantite, PU, heure, mois_num, dow_num, PT, Continent_num]],
                            columns= ['Quantite', 'PU', 'heure', 'month', 'dow', 'PT', 'Continent'])
        pred = model.predict(new_data)[0]
        return pred
    
    if st.button('Lancez la prédiction'):
        prediction = inference(Quantite, PU, heure, mois_num, dow_num, PT, Continent_num)
        if prediction == 0:
            st.success(f"Notre modèle prédit que le client fait partie du cluster 0")
            st.write("Les clients de ce cluster sont des acheteurs réguliers et fidèles")
            st.write("Comme strategie marketing, il est conseillé de fidéliser avec un **programme de fidélité** (réductions cumulées, points, cadeaux), et proposer des ventes croisées (cross-selling) pour augmenter le panier moyen")

        if prediction == 1:
            st.success(f"Notre modèle prédit que le client fait partie du cluster 1")
            st.write("Les clients de ce cluster sont des clients à risque de **churn**")
            st.write("Comme strategie marketing, il est conseillé de relancer avec des **campagnes de réactivation** (emails personnalisés, offres exclusives, codes promo). Mettre en avant la *nouveauté* et les produits tendances pour susciter leur retour. ")

        if prediction == 2:
            st.success(f"Notre modèle prédit que le client fait partie du cluster 2")
            st.write("Les clients de ce cluster sont actifs mais pas encore pleinement fidèles")
            st.write("Comme strategie marketing, il est conseillé d'encourager la **récurrence d’achat** avec des *ventes flash* ou des *abonnements*. Créer une expérience personnalisée (recommandations basées sur achats passés).")

        if prediction == 3:
            st.success(f"Notre modèle prédit que le client fait partie du cluster 3")
            st.write("Les clients de ce cluster sont des **gros dépensiers occasionnels**.")
            st.write("Comme strategie marketing, il est conseillé de mettre en place une **stratégie premium/VIP** (accès anticipé aux nouveautés, service personnalisé). Offrir des avantages exclusifs pour les inciter à acheter plus souvent.")

if selected == "Acceuil":
    Acceuil()

if selected == "EDA":
    EDA_page()

if selected == "Prédiction":
    prediction()


