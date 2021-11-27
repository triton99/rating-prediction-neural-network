# MOVIES RATINGS PREDICTION USING NEURAL NETWORK AND APPICATION IN RECOMMENDATION SYSTEM
<details> 
<summary>Contents</summary>

  - [1. Overview](#1-overview)
    - [Introduction](#introduction)
    - [Project structure](#neural-network-model)
  - [2. Install the required environment and libraries](#2-install-the-required-environment-and-libraries)
  - [3. Running Demo](#3-running-demo)
  - [4. Evaluation](#4-evaluation)

</details>

## 1. Overview
### Introduction
This is my ML subject project in VinBigData Institute with my teammates (Duy Tran, Luc Nguyen, Nam Doan, Tu Phan).
#### Topic
Predicting user ratings for products is very important in recommender systems. Based on this predictive assessment, the system can make effective recommendations, contributing to improving user experience.
There are three main approaches: content-based, collaborative filtering, and matching. This project introduces a neural network that uses a combination of information about users and movie features to make a quantitative assessment of the optimal level of a movie and apply it to a recommendation system.

#### MovieLens20M Dataset
[MovieLens](www.movielens.umn.edu) is a website devoted to researching the recommendation system that was launched in the fall of 1997. Every week hundreds of people visit MovieLens, they give reviews and get recommended movies in return from the system. Load data sets from [MovieLens](www.movielens.umn.edu) and save it in the folder `ml-20m`
This dataset includes the following files:
1. `genome-tags.csv`: genome tag names based on id
2. `movies.csv`: description of movie information, including id, name, genre
3. `tags.csv`: genome tag description for movies
4. `ratings.csv`: user rating description
5. `genome-scores.csv`: describe film characteristics by genre

#### Neural network model
The structure of the proposed neural network:
![The structure of the proposed neural network](/images/model.png)


## 2. Install the required environment and libraries
This project is installed on a virtual environment created by `miniconda`. Select the appropriate version and install `miniconda` [here](https://docs.conda.io/en/latest/miniconda.html).

Initialize virtual environment using `miniconda` with python 3.7:
> `conda create --name <ml-project> python==3.7`

The libraries used in this project are listed in the file [requirements.txt](#requirements.txt):
- `keras==2.6.0, tensorflow==2.7.0, tensorflow_cpu==2.6.1`
- `numpy==1.21.2`
- `pandas==1.1.5`
- `prettytable==2.4.0`
- `scikit_learn==1.0.1`
- `surprise==0.1`

To install the above libraries, execute the following command:
>`pip install -r requirements.txt`

## 3. Running demo
Clone this repo and go to the home directory:
> `git clone https://github.com/triton99/rating-prediction-neural-network.git`

> `cd rating-prediction-neural-network`

> `conda activate <env-name>`
### Predict rating
To run a demo, first you need to download the file [checkpoint](https://drive.google.com/drive/folders/1iGjg6C3ws9QkJL_F_2otydPtpWWEhwC3?usp=sharing) 
save model weights and put in folder `model_checkpoint`.

Next, download the user_genome and movie_genome data here [data](https://drive.google.com/drive/folders/13TyGwiizGIrYNhosUYfVp04J9LX30rsf?usp=sharing) and put in folder `data`.

To predict user rating `u` for movie `i`, execute python source file `rating_predict` :
> `python src/rating_predict.py` 

Then a list of users will be printed, note that this is only the id of the first 100 users, you can try with other user ids.

Enter the id of the user and the id of the movie you want to predict the rating for. The prediction results will be printed to the terminal after some time of execution. The obtained results have the following form:
```
Enter user id: 64
Enter movie id: 1000
-----------------------------------------
Movie name:  Curdled (1996)
Movie genre:  Crime
Actual rating:  None
Movie genome [[0.03475 0.0395  0.0145  ... 0.0075  0.09475 0.01825]]
User genome [[0.03924 0.03645 0.11525 ... 0.03097 0.07641 0.01837]]
-----------------------------------------
Predicted rating:  3.279511
```

### Recommend top K movies

To make a `k` movie recommendation for a user with any `user id`, make sure you have downloaded the checkpoint file saved to the `model_checkpoint` folder and the csv file contains the user's genome data as well as the movies into the folder `data`.

Execute the python source file `topk_predict.py` with the following command:
> `python src/topk_predict.py`

Enter the user code `user_id` and the number of `k` movies you want to recommend:
```
Enter user id: 20000
Enter K: 20
+---------------------------------------------+--------------------------+----------+------------------+--------------+
|                  Movie name                 |          Genre           | Distance | Estimated rating | Ground truth |
+---------------------------------------------+--------------------------+----------+------------------+--------------+
|      Pyromaniac's Love Story, A (1995)      |      Comedy|Romance      |  0.087   |      4.872       |     None     |
|               Babe, The (1992)              |          Drama           |  0.097   |      4.868       |     None     |
|           I'm Not Rappaport (1996)          |          Comedy          |  0.098   |      4.724       |     None     |
|           Robin and Marian (1976)           | Adventure|Drama|Romance  |  0.098   |       4.65       |     None     |
|         Month by the Lake, A (1995)         |   Comedy|Drama|Romance   |  0.099   |      4.649       |     None     |
|            Sandpiper, The (1965)            |      Drama|Romance       |  0.099   |      4.646       |     None     |
|                 Venus (2006)                |      Drama|Romance       |   0.1    |      4.591       |     None     |
|              Life Itself (2014)             |       Documentary        |   0.1    |       4.59       |     None     |
|             Natural, The (1984)             |          Drama           |  0.104   |      4.589       |     None     |
|        Only Angels Have Wings (1939)        | Adventure|Drama|Romance  |  0.105   |      4.528       |     None     |
|            Heaven Can Wait (1978)           |          Comedy          |  0.105   |      4.527       |     None     |
|         Same Time, Next Year (1978)         |   Comedy|Drama|Romance   |  0.108   |      4.497       |     None     |
|               Silverado (1985)              |      Action|Western      |  0.108   |      4.361       |     None     |
| Love Me If You Dare (Jeux d'enfants) (2003) |      Drama|Romance       |  0.109   |      4.352       |     None     |
|          Unfinished Life, An (2005)         |          Drama           |  0.109   |      4.298       |     None     |
|             Away from Her (2006)            |          Drama           |  0.109   |      4.298       |     None     |
|           Wedding Gift, The (1994)          |      Drama|Romance       |  0.109   |      4.247       |     None     |
|                 Wings (1927)                | Action|Drama|Romance|War |   0.11   |      4.246       |     None     |
|          Great Expectations (1946)          |          Drama           |   0.11   |       4.22       |     None     |
|                Ragtime (1981)               |          Drama           |   0.11   |      4.215       |     None     |
+---------------------------------------------+--------------------------+----------+------------------+--------------+
```

## 4. Evaluation
- MAE: Evaluate the absolute value between the predicted rating and the actual rating.
- RMSE: Evaluate the square root of the error of the predicted rating and the actual rating. 
- Two parameters MAE and RMSE are obtained through training using file notebook\compare_SVD_KNN.ipynb`.

| Model    | SVD-Surprise | KNN-Surprise | Neural network |
| ---------| -----------  | -----------  | -------------- |
| MAE      | 0.6384       | 0.6772       | 0.6279         |
| RMSE     | 0.8262       | 0.8739       | 0.8163         |

- Cummulative Hit Ratio: _hit_ rate of the recommender model, assessing the quality of the KNN model (k=20). This parameter is calculated in the file `notebook\KNN_hitrate.ipynb` using the Leave-one-out method.

| Model   | Split 1  | Split 2  | Mean     |
| ------- | -------- | -------- | -------- | 
| CHR     | 0.024315 | 0.023758 | 0.24     |
