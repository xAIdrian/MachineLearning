### Clustering for dataset exploration

#### iris.py

I'm exploring clustering of 2D data points. From an initial scatter plot, it was apparent that the points could be grouped into three distinct clusters. Leveraging this insight, I implemented a KMeans model to identify these clusters and fit it to our data. 

The existing array 'points' from the prior analysis, along with an additional array 'new_points', were used. 

Let's take a closer look at the clustering I performed! I have 'new_points' as an array of points and 'labels' as an array of their cluster labels.

![download](https://github.com/xAIdrian/MachineLearning/assets/7444521/74632aca-5b9e-481d-b01f-ea3d0fc46899)

#### grain.py

The objective here is to identify an appropriate number of clusters for a grain dataset using the k-means inertia graph. This dataset, sourced from the UCI Machine Learning Repository, provides an array of measurements such as area, perimeter, length, and various other attributes of grain samples. With the aid of KMeans and PyPlot, an effective number of clusters will be determined to classify these grain samples, thereby offering a coherent understanding of the inherent patterns and groupings in the data. This methodology leverages the inherent structure of the data, and the output will provide a clear indication of how grain samples can be efficiently categorized based on their measured characteristics.

![download](https://github.com/xAIdrian/MachineLearning/assets/7444521/6b7a518b-9091-423a-b031-5637342ee09e)

The study progresses to evaluate the clustering of the grain samples. An inertia plot was used in the prior analysis, which suggested that three clusters would be optimal for this grain data. Interestingly, these grain samples are a mix from three different varieties: "Kama", "Rosa", and "Canadian". This part of the investigation involves the application of a three-cluster solution to the grain samples and a subsequent comparison of these clusters against the actual grain varieties. The comparison is facilitated through a cross-tabulation technique. The relevant data consists of an array of grain samples and a list detailing the specific variety for each of these samples. The Python libraries utilized in this task are Pandas and KMeans. The objective is to scrutinize the efficacy of the clustering in accurately distinguishing between the different grain varieties.

```
varieties  Canadian wheat  Kama wheat  Rosa wheat
labels                                           
0                       0           1          60
1                      68           9           0
2                       2          60          10
```

#### fish.py

The analysis concerns an array of fish measurements where each row corresponds to an individual fish. The collected data, including weight in grams, length in centimeters, and the percentage ratio of height to length, exhibit varying scales. To enable effective clustering of this data, standardization of these features is a critical prerequisite to ensure that no particular feature dominates the others due to its scale. Therefore, this investigation develops a pipeline for the standardization and clustering of the data. The fish measurement data used in this study were originally published in the Journal of Statistics Education. The aim is to explore the underlying structure and potential groupings in the data by standardizing the features and applying a clustering technique.

The process progresses by employing the previously established standardization and clustering pipeline to group the fish based on their measurements. Post clustering, a cross-tabulation is created to compare the derived cluster labels against the actual fish species. This step helps to assess the accuracy and effectiveness of the clustering process in differentiating among various fish species. The data input includes a 2D array of fish measurements named `samples`, and a list `species` that enumerates the species of each fish sample. Utilizing the `pipeline` for standardization and clustering, the clustering outcomes are then contrasted with the true species to evaluate the level of alignment and discrepancy, thereby offering insights into the validity of the applied methodology.

```
species  Bream  Pike  Roach  Smelt
labels                            
0            0    17      0      0
1           33     0      1      0
2            0     0      0     13
3            1     0     19      1
```

#### stocks.py

This part of the analysis clusters companies based on daily stock price movements from 2010 to 2015. The data is normalized first to account for differences in the scale of stock prices. Unlike StandardScaler, which standardizes features, Normalizer rescales each sample independently. KMeans clustering is then applied to the normalized data to discover potential groupings among companies based on their stock market dynamics.

To determine which companies have similar price change patterns, the cluster labels assigned by the KMeans model are inspected. These labels were generated from a pipeline applied to a NumPy array, `movements`, representing daily stock movements. The list `companies` provides the names of these companies.

```
    labels                           companies
59       0                               Yahoo
15       0                                Ford
35       0                            Navistar
26       1                      JPMorgan Chase
16       1                   General Electrics
58       1                               Xerox
11       1                               Cisco
18       1                       Goldman Sachs
20       1                          Home Depot
5        1                     Bank of America
3        1                    American express
55       1                         Wells Fargo
1        1                                 AIG
38       2                               Pepsi
40       2                      Procter Gamble
28       2                           Coca Cola
27       2                      Kimberly-Clark
9        2                   Colgate-Palmolive
54       3                            Walgreen
36       3                    Northrop Grumman
29       3                     Lookheed Martin
4        3                              Boeing
0        4                               Apple
47       4                            Symantec
33       4                           Microsoft
32       4                                  3M
31       4                           McDonalds
30       4                          MasterCard
50       4  Taiwan Semiconductor Manufacturing
14       4                                Dell
17       4                     Google/Alphabet
24       4                               Intel
23       4                                 IBM
2        4                              Amazon
51       4                   Texas instruments
43       4                                 SAP
45       5                                Sony
48       5                              Toyota
21       5                               Honda
22       5                                  HP
34       5                          Mitsubishi
7        5                               Canon
56       6                            Wal-Mart
57       7                               Exxon
44       7                        Schlumberger
8        7                         Caterpillar
10       7                      ConocoPhillips
12       7                             Chevron
13       7                   DuPont de Nemours
53       7                       Valero Energy
39       8                              Pfizer
41       8                       Philip Morris
25       8                   Johnson & Johnson
49       9                               Total
46       9                      Sanofi-Aventis
37       9                            Novartis
42       9                   Royal Dutch Shell
19       9                     GlaxoSmithKline
52       9                            Unilever
6        9            British American Tobacco
```
