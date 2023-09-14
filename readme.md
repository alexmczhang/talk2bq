### Update on 2023-09-14
- add the chart plot feature
- Run 2nd sample input 'In the 2008/2009 season, who are the top 10 teams in terms of goals scored?'
- Click "Show Bar Plot" button


## Ask BiqQuery

This demo is to showcase answering questions on a tabular data available in Big Query using Vertex PALM LLM & Langchain.

This demo uses a sample public dataset from Kaggle (https://www.kaggle.com/datasets/hugomathien/soccer)

### Sample Inputs:
1. What is short name for FC Barcelona ?
2. In the 2008/2009 season, who are the top 10 teams in terms of goals scored?
3. In the 2008/2009 season, who are the top 10 teams in terms of goals scored, and how many goals they scored at home and away?
4. How many matches FC Barcelona won in the 2008/2009 season as home team ?
5. Here is the rule for each match, win = 3 points, draw = 1 point, lost = 0 point. how many points FC Barcelona had for season 2008/2009

### Enter a search query...


### Prepare the dataset
1. Donwload the dataset from kaggle.com
2. Setup sqlite env on your linux os
3. Export the each table as CSV
4. Upload the CSV to gcs and import to bigquery

## Create partiation and cluster index for table "match" if necessary

CREATE TABLE hello-world-360207.demo_talk2bq.match_clustered 
CLUSTER BY
  season, home_team_api_id
AS (
  SELECT * FROM <project-id>.<dataset-id>.match
);

CREATE TABLE hello-world-360207.demo_talk2bq.match
AS (
  SELECT * FROM <project-id>.<dataset-id>.match_clustered
);



### Docker build
docker build --tag talk2bq .

docker images 

docker ps

docker run -d -p 8080:8080 talk2bq

docker kill [Container ID]

docker tag talk2bq gcr.io/<Project-ID>/talk2bq-alexgcp:v1.0

docker push gcr.io/<Project-ID>/talk2bq-alexgcp:v1.0
