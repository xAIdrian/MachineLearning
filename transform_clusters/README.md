### Clustering for dataset exploration

I'm exploring clustering of 2D data points. From an initial scatter plot, it was apparent that the points could be grouped into three distinct clusters. Leveraging this insight, I implemented a KMeans model to identify these clusters and fit it to our data. 

The existing array 'points' from the prior analysis, along with an additional array 'new_points', were used. 

Let's take a closer look at the clustering I performed! I have 'new_points' as an array of points and 'labels' as an array of their cluster labels.

![download](https://github.com/xAIdrian/MachineLearning/assets/7444521/74632aca-5b9e-481d-b01f-ea3d0fc46899)

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




