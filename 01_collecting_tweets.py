# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:59:28 2021

@author: brenda
"""

# Libraries
import os
import requests
import pandas as pd
import datetime as dt
from dotenv import load_dotenv
from pre_processor import PreProcessor

import warnings
warnings.filterwarnings('ignore')

# loadinng credenttials as environmen variables
load_dotenv('credentials.env', override = True)

# getting twitter credentials
twitter_key = os.environ.get('api_key')
twitter_secret_key = os.environ.get('api_secret_key')
bearer_token = os.environ.get('bearer_token')

# Get current date
today = dt.datetime.today()
#today = today.strftime("%Y-%m-%d %H:%M")
today = today.strftime("%Y%m%d_%H_%M")


def search_tweets(query, bearer_token = bearer_token, next_token = None):    
    """
    Function to request tweets according to a specific query.
    
    Inputs:
        - query: A string that will be used to find tweets.
                 Tweets must match this string to be returned.
        - bearer_token: Security token from Twitter API.
        - next_token: ID of the next page that matches the specified query.
        
    Outputs: Dictionary (json type) with the requested data.  
    """
    
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    
    # end point
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&"

    params = {
        # select specific Tweet fields from each returned Tweet object
        'tweet.fields': 'text,created_at,lang,possibly_sensitive', # public_metrics
        
        # maximum number of search results to be returned (10 - 100)
        'max_results': 100,
        
        # additional data that relate to the originally returned Tweets
        'expansions': 'author_id,referenced_tweets.id,geo.place_id',
        
        # select specific place fields 
        "place.fields": 'country,full_name,name',
        
        # select specific user fields
        "user.fields": 'location',
        
        # get the next page of results.
        "next_token": next_token,
    }
    
    # request
    response = requests.get(url = url, params = params, headers = headers)

    # verify successfull request
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
        
    else:
        return response.json() 
    

def create_dataframes(json_tweets, today):
    """
    Function to create and organize different data into specific data frames.
    
    Inputs:
        - json_tweets: A dictionary with tweets data.
    
    Outputs: 
        - tweets: Pandas dataframe with relevant information about tweets (to
                  further perform text classification).
                  
        - users: Pandas dataframe with users information.
        
        - places (optional): Pandas dataframe about places where users tweeted. If not a 
                  single tweets contains the place where it was tweeted, then
                  this dataframe will not be returned.
    """
        
    # Create users dataframe
    users = pd.json_normalize(json_tweets['includes']['users']).rename(columns = {"id":"user_id"})
    
    # Create df with tweet's data
    tweets = pd.json_normalize(json_tweets['data']).rename(columns = {"id":"tweet_id", 
                                                                      "geo.place_id":"geo_place_id"})
        
    # Get tweet's type
    tweets['type'] = tweets.referenced_tweets.apply(lambda x: x[0]["type"] if type(x) == list else None)
    
    # get referenced tweets ids
    tweets["ref_tweet_id"] = tweets.referenced_tweets.apply(lambda x: x[0]['id']\
                                                            if isinstance(x, list) else x)

        
    # Drop retweeted tweets and tweets with undefined anguage
    tweets = tweets[tweets["lang"] != "und"]
        
    # id to string
    tweets["tweet_id"] = tweets["tweet_id"].astype(str)
        
    # id to string
    users["user_id"] = users["user_id"].astype(str)
        
    # from string to datetime
    tweets["created_at"] = pd.to_datetime(tweets["created_at"], utc = True)
        
    # Not all users enable their location when tweeting, so
    # we need to check if there are available locations for
    # the tweets returned.
    if "places" in json_tweets['includes'].keys():

        # If the field exists, create a dataframe with the corresponding data
        places = pd.json_normalize(json_tweets['includes']['places']).rename(columns = {"id":"geo_place_id"})
            
        # Drop cols
        tweets = tweets.drop(['referenced_tweets','edit_history_tweet_ids'], axis = 1)
        return tweets, users, places
        
    else:
        # Drop cols
        tweets = tweets.drop(['referenced_tweets','edit_history_tweet_ids'], axis = 1)
        return tweets, users
    
# search term
query = "world cup"
search_tweet = search_tweets(query = query)


# Check if we have tweet's location
if "places" in search_tweet['includes'].keys():
    main_tweets, main_users, main_places = create_dataframes(search_tweet, today)
    
else:
    main_tweets, main_users = create_dataframes(search_tweet, today)
    main_places = pd.DataFrame()

for i in range(1, 41):
    
    # Check if there is a next token (another page)
    # that matches the desired query
    if 'next_token' in search_tweet['meta'].keys():
        print(i, search_tweet["meta"]["next_token"])

        # Collect data from next token
        new_tweets = search_tweets(query = query, next_token = search_tweet['meta']['next_token'])
        search_tweet = new_tweets

        # Check if any tweet has enabled the location,
        # so we can create the places dataframe.
        if "places" in search_tweet['includes'].keys():
            tweets, users, places = create_dataframes(search_tweet, today = today)

            # Append data to main tweets
            main_tweets = main_tweets.append(tweets)
            main_users = main_users.append(users)
            main_places = main_places.append(places)

            # Reset index
            main_tweets = main_tweets.reset_index(drop = True)
            main_users = main_users.reset_index(drop = True)
            main_places = main_places.reset_index(drop = True)

        # If any tweet has its location enabled, then only
        # create the other two dataframes.
        else: 
            tweets, users = create_dataframes(search_tweet, today = today)

            # Append data to main tweets
            main_tweets = main_tweets.append(tweets)
            main_users = main_users.append(users)

            # Reset index
            main_tweets = main_tweets.reset_index(drop = True)
            main_users = main_users.reset_index(drop = True)

    # If there are not more results regarding the
    # requested topic, then just stop requesting 
    # more data.
    else:
        break

    
print(f"\n{main_tweets.shape[0]} tweets collected so far")
print(f"{main_users.shape[0]} users collected so far")
print(f"{main_places.shape[0]} user's locations collected so far")


# Store the data locally
if main_places.empty:
    main_tweets.to_csv(f"data/tweets/tweets_{today}.csv", index = False)
    main_users.to_csv(f"data/users/users_{today}.csv", index = False)
    
else:
    main_tweets.to_csv(f"data/tweets/tweets_{today}.csv", index = False)
    main_users.to_csv(f"data/users/users_{today}.csv", index = False)
    main_places.to_csv(f"data/places/places_{today}.csv", index = False)

# update values
main_tweets.loc[main_tweets["possibly_sensitive"] == False, "possibly_sensitive"] = 0
main_tweets.loc[main_tweets["possibly_sensitive"] == True, "possibly_sensitive"] = 1

# get unique tweets ids from referenced tweets that were retweeted
ref_tweets = main_tweets[main_tweets["type"] == "retweeted"].ref_tweet_id.unique()

# get unique tweets ids from original tweets
og_tweets = main_tweets.tweet_id.unique().tolist()

# get unique tweets that have been referenced
# i.e., retweeted, quoted, etc
both = [i for i in ref_tweets if i in og_tweets]

# generate a new dataframe without tweets that have
# referenced other tweets in order to avoid extra
# processing when cleaning them
sample_df = main_tweets[~main_tweets["ref_tweet_id"].isin(og_tweets)]
sample_df = sample_df[["tweet_id", "text", "lang"]].reset_index(drop = True)


# Clean data and only keep the roots of each word.
pre_processor = PreProcessor()
sample_df = pre_processor.lemmatizeWords(sample_df)

# merge main dataframe with sample df to get the clean tweet
main_tweets = main_tweets.merge(sample_df[["tweet_id", "clean_tweet"]],
                                how = "left", on = "tweet_id")
main_tweets = main_tweets.dropna(subset = ["clean_tweet"]).reset_index(drop = True)


if "withheld.copyright" and "withheld.country_codes" in main_tweets.columns:
    main_tweets = main_tweets.drop(["withheld.copyright",
                                    "withheld.country_codes"], axis = 1)


for tweet in both:
    # get the clean tweet from the original tweet
    clean_tweet = str(list(main_tweets.loc[(main_tweets["tweet_id"] == tweet), "clean_tweet"])[0])
    
    # assign the clean tweet to the "clean_tweet" column
    main_tweets.loc[(main_tweets["ref_tweet_id"] == tweet), "clean_tweet"] = clean_tweet

# Store the data locally
main_tweets.to_csv(f'data/clean_tweets/clean_tweets_{today}.csv', index = False)
print("\n\nTweets collection process finished! \N{ghost}")
