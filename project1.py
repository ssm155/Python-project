import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from sklearn.cluster import KMeans

# reading the data
data = pd.read_csv('C:/Users/saram/OneDrive/Desktop/python project/datatest.csv')

# ensuring that all columns are displayed in the details
pd.set_option('display.max_columns',None)
print("THESE ARE THE COLUMNS OF THE TABLE\n")
print(data.head())
print('\n\n\n')

# checking the characteristics of the attributes
print("ANALYSING THE DATA ENTRIES IN THE ATTRIBUTES\n")
print(data.info())
print('\n\n\n')

# removing the '$' and ',' from the income attribute and changint the type to floating point variable
data['income'] = data['income'].str.replace('$', '')
data['income'] = data['income'].str.replace(',', '').astype(float)

# checking for null values in the data entries
print("CHECKING FOR NULL VALUES IN THE ATTRIBUTES\n")
print(data.isnull().sum())
print("\n THE NULL VALUES WERE CHANGED TO THE MEAN OF THE INCOME")
print('\n\n\n')

# changing the entries from null to the mean of the income for the missing values
data['income'] = data['income'].fillna(data['income'].mean())

# changing the entries in age to the mean age, where age >100
print("CHECKING FOR CUSTOMERS WHOSE AGE WAS ENTERED AS GREATER THAN 100\n")
print(data[data['age'] > 100])
a = np.array(data['age'].values.tolist())
data['age'] = np.where(a >100, data['age'].mean(), a).tolist()
print("\n THE MEAN AGE OF THE CUSTOMERS IS\n")
print(data['age'].mean())
print("\n THE AGE VALUES THAT WERE GREATER THAN 100 WERE CHANGED TO THIS MEAN VALUE")
print('\n\n\n')

# exploratory data analysis
# to visualize the customer population according to age
sns.boxplot(data = data,y = 'age',color = 'purple')
plt.show()
print("\n\n\n THE BOXPLOT SHOWS THAT THE AGES OF THE CUSTOMERS IS MOSTLY BETWEEN 44 AND 62")
print('\n\n\n')

#majority of the customers lie between the ages of 44-62 and the mean age of the customers is 52

#visualizing categories of the customer population according to level of education
plt.figure(figsize = (7,7))
sns.countplot(x= data['level_of_ed'], palette = 'rocket_r')
plt.show()
print("THE PLOT SHOWS THAT MOST OF THE CUSTOMERS HAVE MORE THAN A BASIC EDUCATION AND THAT THOSE WITH A GRADUATION DEGREE FORM THE HIGHEST POPULATION OF THIS SAMPLE\n\n\n")

#majority of the customers are of the graduation level, followed by doctorates and master students. The least population is of basic level.

#visualizing categories of the customer population according to marital status
plt.figure(figsize = (7,7))
sns.countplot(x= data['marital_status'],palette = 'rocket_r')
plt.show()
print("CUSTOMERS THAT ARE EITHER MARRIED OR LIVING WITH A SIGNIFICANT OTHER FORM THE MAJORITY OF THE POPULATION\n\n\n")

#ambiguous entries of YOLO, Absurd and Alone, changing all of them to single status
data['marital_status'] = data['marital_status'].apply(lambda x: 'Single' if str(x) in ['YOLO', 'Alone', 'Absurd'] else str(x))
plt.figure(figsize = (7,7))
sns.countplot(x= data['marital_status'],palette = 'rocket_r')
print("THE FEW ENTRIES OF MARITAL STATUS AS YOLO, ALONE AND ABSURD WERE CHANGED THE THE STATUS OF SINGLE, WHICH DID NOT MAKE ANY SIGNIFICANT CHANGES\n\n\n")

#majority of the customers are married or living with a significant other, divorcees and widowers form the smallest group of customers- may also be a reflection of a percentage of the population being so

#visualizing categories of the customer population according to income
plt.figure(figsize = (7,7))
sns.histplot(data['income'])
plt.show()
print("THE MAJORITY OF THE CUSTOMERS HAVE AN INCOME THAT FALLS BELOW 100,000 AND THERE ARE A FEW CUSTOMERS THAT HAVE AN INCOME 1.5- 6 TIMES THIS VALUE\n\n\n")

#the graph shows that there are a few customers with much higher income than the mean of the population, but the majority of the population income lies below 125000

#analyzing relationships between the different atributes

#relationship between age and income
sns.regplot(x= data['age'], y = data['income'])
plt.show()
print("THE PLOT SHOWS THAT THE CUSTOMERS RANGE IN INCOME DOES NOT VARY SIGNIFICANTLY WITH AGE: THE LINEAR RELATION IS VERY SLIGHT\n\n\n")
#not much of a correlation between age and income in this group of customers
#a very slight increase in income with age can be seen

#relationship between income and total purchases
sns.regplot(x = data['income'], y = data['total_purchase'])
plt.xlim([0 , 200000])
plt.show()
print("THE TOTAL PURCHASES FROM THE TWO YEARS IS DIRECTLY PROPORTIONAL TO THE INCOME OF THE CUSTOMERS\n\n\n")

#there is a direct relationship between income and total purchases

#relationship between purchases and dependents
data.groupby(['no_of_dependents'])['total_purchase'].sum().plot()
plt.show()
print("IT IS OBVIOUS THAT WITH AN INCREASE IN NUMBER OF DEPENDENTS, THE TOTAL PURCHASES BY THE CUSTOMER DECREASES. THE PURCHASES FALL RAPIDLY ONCE THERE IS MORE THAN ONE DEPENDENT. FOR FOCUSING ON TOTAL PURCHASING CAPACITY, THOSE CUSTOMERS WITH NO OR ONE DEPENDENT CAN BE SEEN AS THE MAJOR MARKET\n\n\n")

#relationship between purchases and country
data.groupby(['Country'])['total_purchase'].sum().plot()
plt.show()
print("THE GEOGRAPHICAL LOCATION OF THE MOST NUMBER OF CUSTOMERS MAY BE SIGNIFICANT TO STUDY FOR THE PURPOSE OF OPENING A NEW STORE, TO KNOW THE BEHAVIOR OF THE CUSTOMERS, OR TO ESTIMATE SHIPPING COSTS AND UNDERSTAND THE DEMAND OF THE PRODUCTS ACCORDING TO THE COUNTRY\n\n\n")

#relationship between marital status and purchases
data.groupby(['marital_status'])['total_purchase'].sum().plot()
plt.show()
print("THE CUSTOMERS WHO ARE MARRIED OR LIVING WITH A SIGNIFICANT OTHER ARE THOSE WHO CONTRIBUTE TO THE MOST AMOUNT IN PURCHASES. THIS COULD BE RELATED TO THE FACT THAT THE POPULATION OF THESE CATEGORIES IS HIGHER IN THIS SAMPLE. SO, SPENDING HABITS CANNOT BE FIRMLY ESTABLISHED YET\n\n\n")

#amounts of each product purchased over the two years
prod = data[['fruits_qty', 'meat_qty', 'fish_qty', 'sweets_qty', 'gold_qty']].agg([sum]).T
sns.barplot(x = prod.index, y = prod['sum'])
plt.gca().set_xticklabels(['Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'])
plt.xlabel('Products')
plt.ylabel('Amount')
plt.show()
print("THE PURCHASE OF MEAT BROUGHT IN THE MOST REVENUE FOR THE TWO YEARS AT OVER $350,000. ALL THE OTHER CATEGORIES BROUGHT IN BETWEEN $50,000 TO $100,000\n\n\n")

#purchases made through each mode
plt.figure(figsize = (8, 6))
tp = data[['no_of_deals', 'web_purch', 'catalog_purch', 'store_purch']].agg([sum])

sns.barplot(x = tp.T.index, y = tp.T['sum'], palette = 'mako_r')
plt.gca().set_xticklabels(['Deals', 'Web', 'Catalog', 'Store'])
plt.xlabel('Purchase Through')
plt.ylabel('Purchases')
plt.show()
print("THE PLOT SHOWS THAT THE MAJORITY OF THE NUMBER OF PURCHASES WAS DONE IN THE STORE OR ONLINE. THIS SHOWS THAT ENSURING AVAILABILITY OF THE PRODUCTS IN DEMAND IN THE STORE AND ONLINE CAN ENSURE OPTIMUM SALES. SOME KIND OF DISCOUNT WAS AVAILED FOR OVER 5000 PURCHASES. FURTHER RESEARCH INTO THE PRODUCTS FOR WHICH THE DISCOUNTS WERE AVAILED CAN LEAD TO RESULTS THAT SHOW WHICH CUSTOMERS OR PRODUCTS TO TARGET WHILE PROVIDING THE DISCOUNTS\n\n\n")

#moving onto kmeans clustering
#changing the columns from strings/date to numeric values
print("WE HAVE SEEN THE CHARACTERISTICS OF THE CUSTOMERS AND PURCHASES AND CAN NOW CHECK WHETHER THERE ARE ANY GROUPD OF CUSTOMERS THAT CAN BE FORMED FOR THE SAKE OF DEMOGRAPHIC CATEGORIZING OF CUSTOMERS. THIS CAN BE USEFUL FOR AIMNIG MARKETING TECHNIQUES AND ALSO FOR IDENTIFYING MORE PRODUCTS THAT MAY INTEREST THE CUSTOMER. IRRELEVANT PRODUCTS CAN BE DISCONTINUED AND AREAS OF IMPROVEMENT CAN BE IDENTIFIED\n\n\n")
print("KMEANS CLUSTERING IS BEING EMPLOYED IN THE SAMPLE\n\n\n")
print("ALL THE NON-NUMERIC DATA HAS TO BE CONVERTED TO NUMERIC DATA FIRST\n\n\n")
print(data.head(20))
number = LabelEncoder()
data['level_of_ed'] = number.fit_transform(data['level_of_ed'].astype('str'))
data['marital_status'] = number.fit_transform(data['marital_status'].astype('str'))
data['Country'] = number.fit_transform(data['Country'].astype('str'))
data = data.drop(['join_date', 'ID', 'birth_year', 'young_children', 'teens', 'wine_qty', 'fruits_qty', 'meat_qty', 'fish_qty', 'sweets_qty', 'gold_qty', 'no_of_deals', 'web_purch', 'catalog_purch', 'store_purch', 'web_visits', 'Country'], axis= 1)
print(data.head(50))
print("THE FOLLOWING NUMERIC VALUES WERE ASSIGNED TO THE LEVELS OF EDUCATION:\n2ND CYCLE -0\nBASIC - 1\nGRADUATION - 2\nMASTER -3\nPhD - 4\n\n\n")
print("THE FOLLOWING NUMERIC VALUES WERE ASSIGNED TO THE MARITAL STATUSES:\nDIVORCED - 0\nSINGLE - 1\nMARRIED - 2\nTOGETHER - 3\nWIDOWED - 4\n\n\n")

#Scaling the data for making clusters
X = data.values
X = np.nan_to_num(X)
sc = StandardScaler()

#clustering the data
cluster_data = sc.fit_transform(X)
print('Cluster data samples: ',cluster_data[:5])
print("\n\n\n")
print("IN ORDER TO DETERMINE THE OPTIMAL NUMBER OF CLUSTERS TO USE IN THE MODELING, WE USE THE ELBOW METHOD AND FIND THE POINT WHERE THE ELBOW OF THE CURVE IS FORMED. THAT INTEGER IS ACCEPTED AS THE NUMBER OF CLUSTERS IN THE MODEL\n\n\n")

#k-means modeling using three parameters
#implementing elbow method to determing the number of clusters for KMeans modeling
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
print("THE NUMBER OF CLUSTERS IS ADOPTED AS 4, FROM THE FIGURE\n\n\n")
#clustering using KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=12, random_state=0)
y_kmeans = kmeans.fit_predict(X)
labels = kmeans.labels_
print("THE CLUSTERING GIVES THE FOLLOWING RESULT:\n")
print(labels[:100])
print('\n\n\n')

#adding the cluster numbers to the table
print("THE CLUSTER NUMBER CAN BE ADDED TO THE TABLE IN ORDER TO SEE WHICH CLUSTER EACH CUSTOMER BELONGS TO\n")
data['cluster_num'] = labels
print(data.head())
print('\n\n\n')

#finding trends in the clusters
print("THE MEAN VALUES OF THE ATTRIBUTES FOR THE CLUSTERS IS AS SHOWN\n")
print(data.groupby('cluster_num').mean())
print("FROM THE TABLE, WE CAN SEE THAT THERE ARE FOUR CLUSTERS THAT ARE FORMED AND WITH THE CLUSTERS, THE CUSTOMERS CAN BE CATEGORIZED AS:\n 1.CUSTOMERS WITH NO DEPENDENTS AND AVERAGE INCOME OF $76,000 IS THE GROUP THAT HAS THE HIGHEST AVERAGE TOTAL PURCHASE \n 2. CUSTOMERS WITH 1 OR MORE DEPENDENTS AND A SLIGHTLY LOWER AVERAGE INCOME OF $52,000 FORMS THE NEXT GROUP WITH AVERAGE TOTAL PURCHASES OF $500 \n 3. THIS IS THE A SINGLE DATA ENTRY THAT HAD AN INCOME MUCH HIGHER THAN THE REST OF THE CUSTOMERS, SO CANNOT BE REGARDED AS A CLUSTER\n 4. THE FINAL CLUSTER IS OF CUSTOMERS OF A SLIGHTLY LOWER AGE GROUP WITH 1 OR MORE DEPENDENTS AND LOWER INCOME WITH A TOTAL NUMBER OF 94 PURCHASES\n\n")
 
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()

print("IN ORDER TO CHECK THE CONSISTENCY OF THE CLUSTERS, DROPPING THE COLUMN OF EDUCATION LEVEL AND RECLUSTERING FOR ANALYSIS OF THE RESULT\n\n\n")

#dropping columns to see if there is a change in the clusters
data = data.drop(['level_of_ed'], axis = 1)
#repeat clustering
#Scaling the data for making clusters
X = data.values
X = np.nan_to_num(X)
sc = StandardScaler()

#clustering the data
cluster_data = sc.fit_transform(X)
print('Cluster data samples: ',cluster_data[:5])
print("IMPLEMENTING THE ELBOW METHOD AGAIN\n\n\n")

#k-means modeling using three parameters
#implementing elbow method to determing the number of clusters for KMeans modeling
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#clustering using KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=12, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
print(labels[:100])

print("THE CLUSTERED DATA IS AS FOLLOWS:\n")
#adding the cluster numbers to the table
data['cluster_num'] = labels
print(data.head())

#finding trends in the clusters
print("\n THE MEAN VALUES FOR THE CLUSTERS ARE \n")
print(data.groupby('cluster_num').mean())
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()
print("\n\n\n")

print("REPEATING AFTER DROPPING THE ATTRIBUTE 'MARITAL STATUS'\n\n\n")

data = data.drop(['marital_status'], axis = 1)
#repeat clustering
#Scaling the data for making clusters
X = data.values
X = np.nan_to_num(X)
sc = StandardScaler()

#clustering the data
cluster_data = sc.fit_transform(X)
print('Cluster data samples: ',cluster_data[:5])

#k-means modeling using three parameters
#implementing elbow method to determing the number of clusters for KMeans modeling
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#clustering using KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=12, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
print(labels[:100])

#adding the cluster numbers to the table
print("THE CLUSTERED DATA IS AS FOLLOWS\n")
data['cluster_num'] = labels
print(data.head())
print("\n THE MEAN VALUES FOR THE CLUSTERS ARE\n")
#finding trends in the clusters
print(data.groupby('cluster_num').mean())
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()
print('\n\n\n')
print("REPEATING CLUSTERING AFTER REMOVING THE ATTRIBUTE 'NUMBER OF DEPENDENTS'\n")
#removing number of dependents and repeating clustering
data = data.drop(['no_of_dependents'], axis = 1)
#repeat clustering
#Scaling the data for making clusters
X = data.values
X = np.nan_to_num(X)
sc = StandardScaler()

#clustering the data
cluster_data = sc.fit_transform(X)
print('Cluster data samples: ',cluster_data[:5])

#k-means modeling using three parameters
#implementing elbow method to determing the number of clusters for KMeans modeling
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#clustering using KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=12, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
print(labels[:100])

#adding the cluster numbers to the table
print("THE RESULTS AFTER CLUSTERING ARE AS FOLLOWS:\n")
data['cluster_num'] = labels
print(data.head())
print("THE MEAN CALUES FOR EACH CLUSTER ARE AS FOLLOWS:\n")
#finding trends in the clusters
print(data.groupby('cluster_num').mean())
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()
print("\n\n\n REPEATING CLUSTERING AFTER DROPPING THE ATTRIBUTE 'AGE' \n")
#removing age
data = data.drop(['age'], axis = 1)
#repeat clustering
#Scaling the data for making clusters
X = data.values
X = np.nan_to_num(X)
sc = StandardScaler()

#clustering the data
cluster_data = sc.fit_transform(X)
print('Cluster data samples: ',cluster_data[:5])

#k-means modeling using three parameters
#implementing elbow method to determing the number of clusters for KMeans modeling
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#clustering using KMeans
kmeans = KMeans(n_clusters=4
                , init='k-means++', max_iter=300, n_init=12, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
print(labels[:100])

#adding the cluster numbers to the table
print("THE RESULTS OF CLUSTERING ARE AS FOLLOWS: \n")
data['cluster_num'] = labels
print(data.head())

#finding trends in the clusters
print("THE MEAN VALUES FOR THE CLUSTERS ARE AS FOLLOWS: \n")
print(data.groupby('cluster_num').mean())
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.show()
print('\n\n\n')
print("THE RESULTS FROM ALL THE CLUSTERING SHOWS THAT THE CLUSTERING BASED ON INCOME, AGE AND TOTAL PURCHASES HAS BEEN TRULY VERIFIED AND THAT THE CLUSTERS AS DEFINED IN THE FIRST CLUSTERING ARE ACCEPTABLE FOR THE ANALYSIS RESULTS. THERE IS NO MARKABLE DISTINCTION IN THE CUSTOMERS BASED ON AGE AND LEVEL OF EDUCATION. THIS MAY BE BECAUSE THE MAJORITY OF THE CUSTOMERS ARE GRADUATION LEVEL HOLDERS. THE MAJORITY OF THE CUSTOMERS ARE ALSO MARRIED OR LIVING WITH A SIGNIFICANT OTHER. THIS MAY BE THE EXPLANATION FOR THE LACK OF CLUSTERS DEMARKING THIS CATEGORY. THUS, THE CONCLUSION IS THAT CLEAR CLUSTERING IS PREVALENT IN THE CUSTOMER SAMPLE BASED ON INCOME AND THE NUMBER OF DEPENDENTS IN THE HOUSEHOLD IN THE DECREASING ORDER OF TOTAL PURCHASES MADE AS: CUSTOMERS WITH HIGHER INCOME AND NO DEPENDENTS, CUSTOMERS WITH HIGHER INCOME AND ONE DEPENDENT, CUSTOMERS WITH LOWER INCOME AND MORE DEPENDENTS. OTHER CONCLUSIONS FROM THE INITIAL EXPLORATORY ANALYSIS INCLUDE THE FOLLOWING: THE MAJORITY OF THE CUSTOMERS OF THE ENTERPRISE ARE GRADUATES, AND HAVE AN INCOME BETWEEN 0-$100,000. THE CUSTOMERS ARE MOSTLY PERSONS WHO ARE MARRIED OR LIVING WITH A SIGNIFICANT OTHER. THE MOST NUMBER OF PURCHASES WAS FROM SPAIN. THE HIGHER PURCHASES WERE DONE BY CUSTOMERS WITH NONE OR ATMOST ONE DEPENDENT. MOST OF THE PURCHASES WERE CONDUCTED IN THE STORE, FOLLOWD BY ONLINE PURCHASES. AGE IS NOT A RELEVANT FACTOR IN THIS CUSTOMER SAMPLE.")
