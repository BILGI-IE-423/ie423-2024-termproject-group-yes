## Abstract

This study explores the application of two distinct machine learning models to analyze and
optimize film offerings on Netflix for both screenwriters and viewers. The first model assists
screenwriters by providing insights into the average duration of films within specific categories,
aiding in the creation of content that aligns with platform expectations and viewer preferences.
The second model focuses on enhancing the viewing experience by recommending films with
high IMDb scores or those that have won an Oscar, based on the viewer's selected category.
Together, these models aim to improve content creation strategies and personalized
recommendations, ultimately enhancing user satisfaction and engagement on Netflix.

# Scope of the Project

**1 - Data Collection and Preparation:**

Compile a comprehensive dataset of films available on Netflix, including metadata such as
category, duration, IMDb scores, and Oscar wins.

Clean and preprocess the data to ensure accuracy and consistency.

**2 - Model Development:**

Develop a machine learning model to analyze the average duration of films within various
categories, providing actionable insights for screenwriters.

Create a recommendation system using machine learning algorithms to suggest films with high
IMDBscores or Oscar wins based on viewer preferences.

**3 - Analysis and Insights:**

Conduct a thorough analysis of the collected data to identify patterns and trends in film durations
across different categories.

Evaluate the performance of the recommendation system in terms of accuracy and relevance of
suggestions.

**4 - Implementation and Testing:**

Implement the developed models within a user-friendly interface for easy access by screenwriters
and viewers.

Test the models with real user data to refine and improve their accuracy and effectiveness.

**5 - Evaluation and Improvement:**

Continuously monitor and evaluate the models' performance using key metrics such as user
satisfaction, engagement rates, and recommendation relevance.

Incorporate user feedback and make necessary adjustments to enhance the models' capabilities.


By addressing these objectives, the project aims to provide valuable tools for both content
creators and consumers on Netflix, fostering a more tailored and enjoyable viewing experience.

# Dataset Used

Netflix Movies and Tv Shows

Oscar Winners

IMDB Ratings

# Research Questions:

What is the distribution of content types uploaded to Netflix by years?

What is the relationship between Oscar-Winner films and Netflix Availability?

Is there a relationship between IMDB and Netflix?

# Preprocessing Step or the model for screenwriters:

**Filter the Data:**

“netflix_movies = netflix_data[netflix_data['type'] == 'Movie']”

This step filters the netflix_data DataFrame to include only rows where the type column is
'Movie', excluding TV shows.

**Drop Rows with Missing Values:**

“netflix_movies.dropna(subset=['release_year', 'duration', 'listed_in'], inplace=True)”

This step removes rows from the netflix_movies DataFrame that have missing values in any of
the specified columns: release_year, duration, or listed_in. The inplace=True argument modifies
the DataFrame directly without needing to assign it to a new variable.

**Clean and Convert the 'Duration' Column:**

“netflix_movies['duration'] = netflix_movies['duration'].str.replace(' min', '').astype(int)”

This step performs two actions on the duration column:

- **str.replace(' min', '')** : Removes the string ' min' from the values in the duration column,
    leaving only the numeric part.
- **.astype(int)** : Converts the cleaned string values to integers.


**Calculate Average Duration:**

“average_duration = netflix_movies.groupby('listed_in')['duration'].mean().reset_index()

average_duration.columns = ['listed_in', 'avg_duration']”

This step calculates the average duration for each movie category:

- **groupby('listed_in')** : Groups the DataFrame by the listed_in column (which contains the
    categories).
- **['duration'].mean()** : Calculates the mean duration for each group.
- **reset_index()** : Resets the index to get a DataFrame with the grouped column (listed_in)
    and the calculated mean (duration).
- **average_duration.columns = ['listed_in', 'avg_duration']** : Renames the columns for
    clarity.

**Merge Average Duration:**

“netflix_movies = pd.merge(netflix_movies, average_duration, on='listed_in', how='left')”

This step merges the average_duration DataFrame back to the netflix_movies DataFrame:

- **on='listed_in'** : Specifies that the merge should be done based on the listed_in column.
- **how='left'** : Uses a left join to keep all rows from netflix_movies and add the
    corresponding avg_duration values from average_duration.

**One-Hot Encoding for Categories:**

” netflix_movies = pd.get_dummies(netflix_movies, columns=['listed_in'])”

This step converts the listed_in categorical column into multiple binary (0 or 1) columns, each
representing a category. This process is known as one-hot encoding.

**Prepare Features and Target Variable:**

“ X = netflix_movies[['release_year', 'avg_duration'] +
list(netflix_movies.columns[netflix_movies.columns.str.startswith('listed_in_')])]

y = netflix_movies['duration']” This step prepares the feature matrix X and the target variable y:

- **X** : Includes the release_year and avg_duration columns, as well as all columns that start
    with listed_in_ (resulting from one-hot encoding).
- **y** : The target variable is set to the duration column, which contains the movie durations.


1. Filter Data: Select only movies.
2. Handle Missing Values: Remove rows with missing release_year, duration, or listed_in.
3. Clean duration: Remove ' min' and convert to integers.
4. Calculate Average Duration: Group by category and calculate the mean duration.
5. Merge Average Duration: Add the calculated average duration back to the original data
    frame.
6. One-Hot Encoding: Convert categorical listed_in column to multiple binary columns.
7. Prepare Features and Target: Select relevant columns for features and set the target
    variable.

**Merging netflix_titles with Oscar data frame** :

This step combines the netflix_titles data frame with the oscar data frame. Here are the key
points:

“netflix_titles_oscar = pd.merge(netflix_titles, oscar, left_on='title', right_on='Film', how='left')”

- **left_on='title'** : This specifies that the title column in the netflix_titles data frame should
    be matched with the Film column in the oscar data frame.
- **right_on='Film'** : This specifies that the Film column in the oscar data frame should be
    used for the merge.
- **how='left'** : This indicates a left join, meaning all rows from the netflix_titles data frame
    will be included in the merged data frame. If a title in netflix_titles doesn't have a
    corresponding film in the oscar data frame, the merged data frame will still include that
    row, but with NaN values for the columns from the oscar data frame.

**Creating a New Column Won Oscar** : This step creates a new column called Won Oscar in the
netflix_titles_oscar data frame. Here’s what it does:

“netflix_titles_oscar['Won_Oscar'] = netflix_titles_oscar['Award'].notna()”

- **netflix_titles_oscar['Award'].notna()** : This checks each value in the Award column to
    see if it is not NaN (i.e., it checks if the value exists).
- **netflix_titles_oscar['Won_Oscar'] =** : This assigns the result of the notna() check to a
    new column called Won_Oscar.

**Merge the resulting dataframe with imdb_data** :

“netflix_titles_imdb = pd.merge(netflix_titles_oscar, imdb_data, left_on='title', right_on='Name',
how='left')”

This step merges the netflix_titles_oscar dataframe with the imdb_data dataframe based on the
title column from netflix_titles_oscar and the Name column from imdb_data. Again, the
how='left' parameter ensures that all rows from netflix_titles_oscar are retained, and only
matching rows from imdb_data are included.

**Rename the 'Rating' column to 'IMDB_Score'** :


“ netflix_titles_imdb.rename(columns={'Rating': 'IMDB_Score'}, inplace=True)”

This step renames the Rating column to IMDB_Score for clarity.

**Convert 'IMDB_Score' to numeric and handle errors** :

“netflix_titles_imdb['IMDB_Score'] = pd.to_numeric(netflix_titles_imdb['IMDB_Score'],
errors='coerce')”

This step converts the IMDB_Score column to a numeric data type. If any values cannot be
converted to numeric, they are replaced with NaN (Not a Number).

**Handle NaN values in IMDB_Score by filling them with 0** :

“ netflix_titles_imdb['IMDB_Score'] = netflix_titles_imdb['IMDB_Score'].fillna( 0 )”

This step replaces any NaN values in the IMDB_Score column with 0.

**Filter movies with IMDB score above 7 or won an Oscar** : “filtered_movies =
netflix_titles_imdb.loc[ (netflix_titles_imdb['IMDB_Score'] > 7 ) |
(netflix_titles_imdb['Won_Oscar']) ].copy()”

This step filters the dataframe to include only those movies that have an IMDB score greater than
7 or have won an Oscar. The .copy() method creates a copy of the filtered dataframe to avoid any
potential issues with chained indexing.

**Combine 'description' and 'listed_in' for TF-IDF** :

“ filtered_movies['combined_features'] = filtered_movies['description'].fillna('') + ' ' +
filtered_movies['listed_in'].fillna('')”

This step creates a new column combined_features by concatenating the description and listed_in
columns, filling any NaN values with empty strings before concatenation. This combined text
will be used for TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

# Data Story for the First Model For Screenwriter:Analyzing the Average Duration of Films on Netflix

# Based on Category

In the dynamic world of streaming services, creating content that resonates with the audience is
key. For screenwriters aiming to craft films for Netflix, understanding the typical duration of
films across different genres can provide invaluable guidance. This study leverages machine
learning to analyze and visualize the average duration of films based on their categories, offering
a data-driven approach to scriptwriting.

**Overview of the Analysis**

The primary objective of this model is to inform screenwriters about the average film duration
based on its category. By understanding these trends, screenwriters can tailor their scripts to align
with audience expectations on Netflix.

**Initial Model Performance**
<div align="center">
    <img src="https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/output.png"
</div>
<div align="left">
- **Horizontal Line Trend** : Most predicted values cluster around zero, forming a horizontal
    line. This suggests that the model is predicting nearly the same duration for many movies,
    regardless of their actual durations.


- **Outliers** : There are a few extreme outliers in the predicted values, some very high, others
    very low (even negative), which is not feasible for movie durations.
- **Lack of Diagonal Trend** : Ideally, we would expect the points to align closely along a
    diagonal line where Actual Duration = Predicted Duration. The absence of this trend
    indicates a poor correlation between actual and predicted values.

**Improved Model Performance**
<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/output2.png width=>
</div>
<div align="left">

After comparing the performance of four different models, the one with the highest R-squared
value was selected.

- Cross-Trend: A clear cross-trend in the points indicates that the model establishes a
    better relationship between actual and predicted values.
- Closer to y = x Line: The points are generally closer to the y = x line, indicating more
    accurate predictions.
- Centered Points: Most points are centered within a certain range (approximately 80- 120
    minutes), suggesting that the durations of most movies fall within this range and the
    model performs better here.
- Outliers: There are still a few outliers representing cases where the model failed to
    accurately predict the duration of some movies.

**Model Refinement by Removing Outliers**
<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/output3.png width=>
</div>
<div align="left">

To further refine the model, outliers were detected and removed using the Z-score method.


- **Outlier Removal** : Removing outliers helps in improving the accuracy and robustness of
    the models by eliminating extreme values that might skew the results.
- **Data Splitting** : The data was split into training and test sets again after outlier removal to
    ensure model training and evaluation were performed on clean data.

This model provides valuable insights into the average duration of films based on their categories
on Netflix. By leveraging these findings, screenwriters can make informed decisions about the
length of their films, ensuring they align with viewer expectations and industry trends. The
visualizations and data analysis presented in this study serve as a practical tool for creating
compelling and appropriately timed content for the streaming platform.

# Data Story for the Second Model For Viewer:Recommendation Films on Netflix Based on Category

- TF-IDF Features: The text descriptions of the movies are transformed into numerical
    vectors using TF-IDF. This captures the importance of words in each movie description.


- PCA: PCA is applied to these TF-IDF features to reduce the dimensionality while
    retaining most of the variance, making the data more manageable for the Nearest
    Neighbors algorithm.
- Algorithm: The Nearest Neighbors algorithm is used to find movies similar to a given
    movie based on the reduced PCA features.
- Recommendation Criteria: The system prioritizes movies with high IMDb scores

## (above 7) or those that have won an Oscar.
<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/training%20set.png>
</div>
<div align="left">

This plot shows the Principal Component Analysis (PCA) of movies based on TF-IDF features
for the training set. PCA reduces the high-dimensional feature space (created by TF-IDF) into
two principal components, making it easier to visualize and understand.

<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/test%20set.png>
</div>
<div align="left">
This plot shows the PCA of movies for the test set, projected into the same feature space as the
training set. It illustrates how new, unseen movies are distributed based on the learned PCA
components from the training set.

**Metrics Interpretation**

- **Average Precision** : The value of 1.051851851851852 indicates an issue, as precision
    should be between 0 and 1. This suggests a potential problem with the calculation or the
    labeling of relevant movies.
- **Average Recall** : The value of 5.04320987654321 is also incorrect, as recall should also
    be between 0 and 1. This likely points to an issue with how relevant movies are identified
    in the dataset.

**Evaluating the model**

**Comparison of Correct Categories** :

- By comparing the true categories and recommended categories as sets, the correct
    categories (true positives) and incorrect categories (false positives and false negatives) are
    accurately calculated.

**Calculation of Correct Metrics** :

- Precision and recall metrics are used to measure how accurate and complete the model's
    recommendations are. These metrics play a crucial role in evaluating the model's
    performance.

**Detailed and Systematic Evaluation** :


- Precision and recall are calculated individually for each movie, and the overall averages
    are taken. This allows for a more comprehensive evaluation of the model's performance.

After the model evaluation with an average precision of 0.20987736543292101 and an average
recall of 0.9753086419753088.The model is very good at identifying most of the relevant
categories (high recall), but it also suggests many irrelevant categories (low precision).
<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/evaluated%20traiing%20set.png>
</div>
<div align="left">
<div align="center">
    <img src=https://github.com/BILGI-IE-423/ie423-2024-termproject-group-yes/blob/main/Datasets/evaulated%20test%20set.png>
</div>
<div align="left">

# Future Works


**Expanding to Other Streaming Platforms** :

- Extending the analysis to include other major streaming services such as Amazon Prime,
    Hulu, and Disney+ would provide a broader perspective on content trends and viewer
    preferences across different platforms.

**Enhanced Personalization with Viewer Profiles:**

Incorporating detailed viewer profiles that include viewing history, preferred genres, and
demographic information could refine the recommendation model. This personalized approach
could result in even more tailored and relevant movie suggestions.

Incorporation of Genre-Specific Trends:

- Analyzing genre-specific trends over time, such as the evolution of popular genres, could
    help screenwriters and producers anticipate future trends and tailor their content
    accordingly.

Impact of Release Timing and Marketing:

- Investigating the impact of release timing, promotional strategies, and seasonal effects on
    film performance could offer strategic insights for maximizing viewer engagement and
    content success.

International Market Analysis:

- Conducting a comparative analysis of viewer preferences and content trends across
    different international markets could help tailor content strategies to diverse global
    audiences.

Real-Time Data Integration:

- Implementing real-time data integration could allow for dynamic updates to both models,
    ensuring that recommendations and insights are based on the most current viewer data and
    trends.

Advanced Machine Learning Techniques:

- Utilizing advanced machine learning techniques such as deep learning and reinforcement
    learning could improve model accuracy and adaptability. These techniques could better
    capture complex patterns and interactions within the data.



