<<<<<<< HEAD
# Inferential Linear Regression

Linear regression is often used a predictive tool, and in this use case we don't need many statistical assumptions for it to do an admirable job.  Another common use for linear regression is *inferential*, and for this use case we need to check some statistical properties of the model.

For this sprint, make sure you have the [py-glm](https://github.com/madrury/py-glm) library installed.  This library supports the inferential tools we will use today with a simple interface (python lacks well designed tools for inferential work, so we wrote this one ourselves).  Please take a moment to read the README for this library.

The process for fitting a linear model with the `GLM` library looks like:

```
from glm.glm import GLM
from glm.families import Gaussian

model = GLM(family=Gaussian())
model.fit(X, y)
```

The two parts of this assignment are in no particular order.  Feel free to do them in either order, here's a short description to guide your thoughts:

  - The first part explores the assumptions of regression, and the tools available if these assumptions hold.
  - The second part explores the situations where linear regression fails.  This section requires some skill in using numpy to create example data.


## Part 1: Assumptions for Inferential Regression

When using a linear regression to answer inferential questions, i.e. as a tool
to answer questions about some hypothetical *population* we need to make quite
a few assumptions about the data generating process (i.e. the population we are
studying with the regression). 

- **Independence of the observations.**
- **Constant Conditional Variance of Errors (Homoskedacity).**
- **Normal Distribution of Errors**

Since the inferential results (i.e. the standard errors of the parameter
estimates) of the regression model depend on these statistical assumptions, the
results of the regression model are only correct if our assumptions hold (at
least approximately).

**Note:**  The *predictions* of the linear regression model are not dependent on these assumptions, so if your goals are purely predictive, these assumptions are not strong concerns.

We will be exploring two datasets: `prestige` and `ccard`. Below is a description of the 2 datasets.

* `prestige`:
    - Target is the prestige of a job
    - Dependent variable: `prestige`.
    - Independent variables: `income`, `education`.  There are others, but you may feel free to exclude them for this exercise.

To load the prestige data set into a data frame:
  
  ```python
  import pandas as pd
  url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/carData/Duncan.csv'
  prestige = pd.read_csv(url)
  ```
   
* `ccard`
    - Target is average credit card expenditure.
    - Dependent variable: `AVGEXP`.
    - Independent variables: `AGE`, `INCOME`, `INCOMESQ`, `OWNRENT`.
  
To load the credit card data set into a data frame:

  ```python
  import statsmodels.api as sm
  credit_card = sm.datasets.ccard.load_pandas().data
  ```

1. Explore the datasets with a [scatter_matrix](https://pandas.pydata.org/pandas-docs/stable/visualization.html#scatter-matrix-plot) and a [boxplot](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html).
   
2. Fit a linear regression model to each of the datasets. Print and examine the
summaries of the models.  The summary should report the parameter estimates and
their standard errors.

3. Plot the residuals of the models against the predicted values.  Do these
residuals show any concerning patterns?  If so, how should you deal with them?
   
4. By inspecting the residual plots, which model is more likely to have
**heteroscedastic** residuals? Explain what heteroscedasticity means.

5. What uses of the model would heteroscedasticity (a violation of homoscedasticity) invalidate?

6. One of the most common treatments to reducing heteroscedasticity is to take
the logarithm of the response variable, especially if the conditional distribution of
the response variable is skewed. Take the log of `AVGEXP` in `ccard` data.
Re-fit the model to the logarithm of `AVGEXP`, and re-plot the residuals. 
   
7. To test if the residuals are normally distributed, the common practice is to
use a qq-plot (for quantile-quantile-plot). The Q-Q plot plots the quantile of
the normal distribution against that of the residuals and checks
for alignment of the quantiles.
    
Make qq-plots for the residuals of the `prestige` and `ccard` (before `log`
transform) models (it is assumed you will have to do a bit of research to make
these plots, we've intentionally omitted how to make them).  Apply the `log` transform to `AVGEXP` in
`ccard` and repeat the plot.  What do you observe?

8. The `p_values_` attribute of the model contains the results of applying a z-test to the parameter estimates.  Discuss the following questions with your partner:
  - What assumptions must hold for this z-test to be valid?
  - What is the null hypothesis of this z-test?
  - What is the distribution of the parameter estimates under the null hypothesis?

9. See if you can calculate these p-values by hand, and see if your results match those given by the library.  If you're stuck, feel free to look through the source code of the `py-glm` library to find [where the code that calculates these p-values](https://github.com/madrury/py-glm/blob/f3d6f68b0024c5fab598749d20c758fd2e9ccb6c/glm/glm.py#L305) is located.

10. Give some examples of scientific questions that could be answered by these p-values.  Give some examples of questions that are *not* answered by these p-values.


## Part 2: A Failure Mode for Linear Regression

The least we could ask of a linear regression is that we can actually fit the model.  It turns out that linear regression has a simple failure mode that can be completely described.

1. Create a feature matrix `X` with two columns and 100 rows.  The first column should be an intercept column of all `1.0`'s, and the second should be randomly sampled from any distribution (a uniform is fine).

2. Create a target vector from a linear data generating process.  For example:

```
y = 1.0 + 2.0 * X[:, 1] + np.random.normal(size=100)
```

3. Fit a linear regression to `(X, y)` data.  Look at the fit coefficients (i.e. the *parameter estimates* in statistical language).  Are they what you expect them to be?  If you had fit the model to 1,000,000 data points, what would change about them? 

4. Create a new feature matrix `X` with three columns and 100 rows.  Make the first two columns the same as your previous `X`, but make the third column a *copy of the second column*, i.e., `X` should have the *same data* in the second and third column.

5. Fit a linear regression to the new `(X, y)` data (`y` should be the same as it was in the previous example).  What happened?

6. Hopefully you got a `LinearAlgError`, so there's something unfortunate going on here.
Think about what you think the correct answer should be, what coefficients *should* the model return?

7. Create a new feature matrix where one column is a multiple of another, and fit a linear regression again, what happened this time?  How can you explain it?

8. Create one last feature matrix where one column is a *linear combination* of two or more other columns.  Fit a linear regression using it.  What happened this time?  Can you explain it?

9. Hopefully you've seen a few linear regressions fail at this point.  Why did they fail?  What is the failure mode for linear regression?
=======
# Spark Dataframes and SparkSQL

## Basic

### Part 1: Initiating a `SparkSession`

1\. Initiate a `SparkSession`. A `SparkSession` initializes both a `SparkContext` and a `SQLContext` to use RDD-based and DataFrame-based functionalities of Spark. If you launched a notebook using `bash scripts/jupyspark.sh`, the SparkSession and SparkContext will already be defined as `spark` and `sc`, respectively.

```python
import pyspark as ps
spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
sc = spark.sparkContext
```

### Part 2: Introduction to SparkSQL

SparkSQL allows you to execute relational queries on **structured** data using
Spark. Today we'll get some practice with this by running some queries on a
Yelp dataset. To begin, you will load data into a Spark `DataFrame`, which can
then be queried as a SQL table.

1\. Load the Yelp business data using the function `.read.json()` from the `SparkSession()` object, with input file `data/yelp_academic_dataset_business.json.gz`.

2\. Print the schema and register the `yelp_business_df` as a temporary
table named `yelp_business` (use the `createOrReplaceTempView` method on your dataframe; this will enable us to query the table later using
our `SparkSession()` object).

Now, you can run SQL queries on the `yelp_business` table. For example:

```python
result = spark.sql("SELECT name, city, state, stars FROM yelp_business LIMIT 10")
result.show()
```

3\. Write a query or a sequence of transformations that returns the `name` of entries that fulfill the following
conditions:

   - Rated at 5 `stars`
   - In the `city` of Phoenix
   - Accepts credit card (Reference the `'Accepts Credit Card'` field by
   ``` attributes.`Accepts Credit Cards` ```.  **NOTE**: We are actually looking for the value `'true'`, not the boolean value True!)
   - Contains Restaurants in the `categories` array.  

   > Hint 1 : `LATERAL VIEW explode()` can be used to access the individual elements
   of an array (i.e. the `categories` array). For reference, you can see the
   [first example](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+LateralView) on this page.

   > Hint 2: In spark, while using `filter()` or `where()`, you can create a condition that tests if a column, made of an array, contains a given value. The functions is [pyspark.sql.functions.array_contains](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.array_contains).

## Advanced

### Part 3: Spark and SparkSQL in Practice

Now that we have a basic knowledge of how SparkSQL works, let's try dealing with a real-life scenario where some data manipulation/cleaning is required before we can query the data with SparkSQL. We will be using a dataset of user information and a data set of purchases that our users have made. We'll be cleaning the data in a regular Spark RDD before querying it with SparkSQL.

1\. Load a dataframe `users` from `data/users.txt` instead using `spark.read.csv` with the following parameters: no headers, use separator `";"`, and infer the schema of the underlying data (for now). Use `.show(5)` and `.printSchema()` to check the result.

2\. Create a schema for this dataset using proper names and types for the columns, using types from the `pyspark.sql.types` module (see lecture). Use that schema to read the `users` dataframe again and use `.printSchema()` to check the result.

Note: Each row in the `users` file represents the user with his/her `user_id, name, email, phone`.

3\. Load an RDD `transactions_rdd` from `data/transactions.txt` instead using `sc.textFile`. Use `.take(5)` to check the result.

Use `.map()` to split those csv-like lines, to strip the dollar sign on the second column, and to cast each column to its proper type.

4\. Create a schema for this dataset using proper names and types for the columns, using types from the `pyspark.sql.types` module (see lecture). Use that schema to convert `transactions_rdd` into a dataframe `transactions`  and use `.show(5)` and `.printSchema()` to check the result.

Each row in the `transactions` file has the columns  `user_id, amount_paid, date`.

5\. Write a sequence of transformations or a SQL query that returns the names and the amount paid for the users with the **top 10** transaction amounts.
>>>>>>> 9a76729ee2ecf91223c4196d28e8a2ea7b959bf1
