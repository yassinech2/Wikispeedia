import math

import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from numpy.random import default_rng
from plotly.subplots import make_subplots
from sentence_transformers.util import cos_sim

# We define the constatns for the analysis
OUT_DEGREE = "out_degree"
CLOSENESS = "closeness"
PAGERANK = "pagerank"
BETWEENNESS = "betweenness"
SEMANTIC = "sematic similarity"
CENTRALITY_MEASURES = [OUT_DEGREE, CLOSENESS, PAGERANK, BETWEENNESS]

K = 12


# We define the data schema for the analysis
class DataSchema:
    """This class defines the data schema for the analysis
    """

    START = "start"
    TARGET = "target"
    PATH = "path"
    VISITS = "visits"
    DURATION = "durationInSec"
    PATH_LENGTH = "path_length"
    TIME_PER_ARTICLE = "time_per_article"
    TYPE = "type"


# for a given series of visited pages, this method removes the nodes that are not in the final path in O(n)
# by folding the back characters ("<") with the page preceding it and removing them from the path
def remove_backs(l):
    # Sanity check on the number of backs performed in the path
    num_backs = len([a for a in l if a == "<"])
    num_pages = len(l) - num_backs
    assert num_backs <= num_pages
    # the idea here is to consider the reverse of the path
    # and count the number of successive backs
    # skip the elements of the path that should not be considered
    l.reverse()
    res = []
    n_to_skip = 0
    i = 0
    while i < len(l):
        elem = l[i]
        if elem == "<":
            n_to_skip += 1
        if n_to_skip == 0:
            res.append(elem)
        elif elem != "<":
            i = i + n_to_skip - 1
            n_to_skip = 0
        i += 1
    res.reverse()
    return res


# this function returns the values of y_s and their relative times (considering the time) that are upper and lower bounding
# the times corresponding to x is a sorted list of values
# x : between 0 and 1
# y_s: a list of values
# returns:
# the xs returned are  between 0 and 1, and y is the uppeer and lower bound of the values of y_s
def get_bucket(y_s, x):
    x_s = [i / (len(y_s) - 1) for i in range(len(y_s))]  # standardize the times
    upper_arr = x_s[1:]
    lower_arr = x_s[:-1]
    i = binary_search(lower_arr, upper_arr, 0, len(lower_arr) - 1, x)
    assert i >= 0
    y1, x1, y2, x2 = y_s[i], x_s[i], y_s[i + 1], x_s[i + 1]
    return y1, x1, y2, x2


# Performs a binary search to find the index i that satisfies lower_arr[i]=< x <= upper_arr[i]
def binary_search(lower_arr, upper_arr, low, high, x):
    if high >= low:
        mid = (high + low) // 2
        if x >= lower_arr[mid] and x <= upper_arr[mid]:
            return mid
        elif lower_arr[mid] > x:
            return binary_search(lower_arr, upper_arr, low, mid - 1, x)
        else:
            return binary_search(lower_arr, upper_arr, mid + 1, high, x)
    else:
        return -1


# returns an estimation of the inbetween value that we will get in the "game" sequence
# by doing a linear estimation
def get_estimation(game, x):
    y1, x1, y2, x2 = get_bucket(game, x)
    return (x - x1) * (y2 - y1) / (x2 - x1) + y1


# Method to calculate the evolution of a particular value
# by averaging over the multiple instances
# it takes an array of a series of values where each series can have a different length (but must be >1)
# and returns an array of length the given number of samples representing the evolution of the average of these values
# as well as the corresponding confidence intervals (sampled with a bootstrapping method) and the time component of each value
def get_value_evolution(games_path_values, num_samples=100):
    points_x = []
    points_estimation = []
    points_upper_bound = []
    points_lower_bound = []
    for x in range(num_samples):
        x = x / num_samples
        points_x.append(x)
        estimations = []
        for game in games_path_values:
            estimations.append(get_estimation(game, x))
        (ci_low, ci_up), mean = get_CI_and_average(estimations)
        points_estimation.append(mean)
        points_lower_bound.append(ci_low)
        points_upper_bound.append(ci_up)
    return points_x, points_estimation, points_lower_bound, points_upper_bound


# returns the mean and the confidence interval of a sample using a bootstrapping method
def get_CI_and_average(X, nbr_draws=1000, confidence=0.95):
    rng = default_rng()
    values = [rng.choice(X, size=len(X), replace=True).mean() for i in range(nbr_draws)]

    CI = np.percentile(
        values, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)]
    )
    average = np.mean(values)

    return (CI, average)


# Method to reduce the Topics
def reduce(s):
    s = s[::-1]
    index1 = s.find(".")
    return s[0:index1][::-1]


# This function will compute similarities between all page hops and the target article using transformers cos_sim
def compute_cosine_similarity(df_embeddings, list_of_articles, target_article):
    df_path = df_embeddings[
        df_embeddings.file_name.apply(
            lambda x: x in list_of_articles + [target_article]
        )
    ]
    target_vector = df_path[df_path.file_name == target_article]["embeddings"].iloc[0]
    list_of_vectors = [
        df_path[df_path.file_name == article]["embeddings"].iloc[0]
        for article in list_of_articles
    ]
    return cos_sim(
        np.array(list_of_vectors).astype(float), np.array(target_vector).astype(float)
    ).numpy()


# Function to fetch summary from Wikipedia API
def get_summary(title, num_sentences=4):
    # Fetch summary from Wikipedia API
    params = {
        "format": "json",
        "action": "query",
        "prop": "extracts",
        "exintro": "",
        "explaintext": "",
        "titles": title,
        "redirects": 1,
        "formatversion": 2,
    }
    api_url = "https://en.wikipedia.org/w/api.php"
    response = requests.get(api_url, params=params).json()
    pages = response["query"]["pages"]
    page = pages[0]
    try:
        summary = page["extract"]
    except:
        summary = "No summary available"
        return summary

    # Tokenize summary into sentences
    sentences = nltk.sent_tokenize(summary)

    # Select first num_sentences sentences
    summary = " ".join(sentences[:num_sentences])

    return summary


# Find to which percent of times your time belongs to
def get_percent(times, your_time):
    # Sort times in ascending order
    times.append(your_time)
    sorted_times = sorted(times)
    # Find index of your time in sorted array
    index = sorted_times.index(your_time)
    # Calculate percent of times that you belong to
    percent = (index + 1) / len(times) * 100
    return np.ceil(percent)


# We define a function to get the k most played games
def get_top_k_games(df_paths_finished, k):
    df_top_k = (
        df_paths_finished[[DataSchema.START, DataSchema.TARGET, DataSchema.PATH]]
        .groupby([DataSchema.START, DataSchema.TARGET])
        .count()
        .sort_values(by=DataSchema.PATH, axis=0, ascending=False)[:k]
    )
    top_k_games = df_top_k.index
    return top_k_games


# Function that returns the average evolution of a measure for the top k most played games
def get_values_plot(df_paths_finished, k, measure, df_nodes=None, df_embeddings=None):
    top_k_games = get_top_k_games(df_paths_finished, k)
    mapping = {}
    for i, game in enumerate(top_k_games):
        path_measure = []

        df_games = df_paths_finished[
            [DataSchema.START, DataSchema.TARGET, DataSchema.PATH]
        ]
        mask = (df_games[DataSchema.START] == game[0]) & (
            df_games[DataSchema.TARGET] == game[1]
        )
        df_games = df_games[mask]
        num_games = df_games.shape[0]

        if measure in CENTRALITY_MEASURES:
            path_measure = list(
                df_games[DataSchema.PATH].apply(
                    lambda l: [df_nodes.loc[elem][measure] for elem in l]
                )
            )
        else:
            paths = df_games[DataSchema.PATH].values
            for path in paths:
                list_of_articles = path[:-1]
                target_article = path[-1]
                path_measure.append(
                    compute_cosine_similarity(
                        df_embeddings, list_of_articles, target_article
                    )
                )

        xs, ys, l, u = get_value_evolution(path_measure)
        mapping[game] = (xs, ys, l, u, num_games)

    return mapping


# Function that plots the average evolution of a measure for the top k most played games
def plot_average_evolution(values, measure):
    num_rows = math.floor(math.sqrt(len(values)))
    num_cols = len(values) // num_rows
    num_cols = num_cols if len(values) % num_rows == 0 else num_cols + 1

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[
            "<b>{} to {} <br> ({} games)</b>".format(key[0], key[1], value[4])
            for key, value in values.items()
        ],
        horizontal_spacing=0.15,
    )

    for i, (key, value) in enumerate(values.items()):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.update_xaxes(
            title_text="path proportion",
            row=row,
            col=col,
            tickfont_size=9,
            titlefont_size=10,
        )
        fig.update_yaxes(
            title_text="{} measure".format(measure),
            row=row,
            col=col,
            tickfont_size=9,
            titlefont_size=10,
            ticksuffix=" ",
        )
        fig.add_trace(
            go.Scatter(x=value[0], y=value[1], mode="lines", showlegend=False),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=value[0], y=value[2], mode="none", fill="tonexty", showlegend=False
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=value[0], y=value[3], mode="none", fill="tonexty", showlegend=False
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=900,
        width=1200,
        title_text="<b> Evolution of {} measure on players' paths \b ".format(
            str(measure)
        ),
        title_x=0.5,
    )
    fig.update_annotations(font_size=13)

    fig.show(renderer='svg')


# Function that yields shortest path from start to end
def get_shortest_path(G, start, end):
    return nx.shortest_path(G, start, end)


# Function that yields evolution of centrality measure over shortest path for top k most played games
def get_centrality_shortest_path(df_paths_finished, G_articles, df_nodes, measure, k=K):
    top_k_games = get_top_k_games(df_paths_finished, k)
    names = []
    for i in top_k_games:
        names.append("{} to {}".format(i[0], i[1]))
    shortest_paths = [
        get_shortest_path(G_articles, game[0], game[1]) for game in top_k_games
    ]
    path_measure = [
        [df_nodes.loc[elem][measure] for elem in path] for path in shortest_paths
    ]
    x = []
    for measure in path_measure:
        x.append(list(range(len(measure))))
    return list(zip(names, x, path_measure))


# Function that plots the evolution of centrality measure over shortest path for top k most played games
def plot_evolution_shortest_path(values, measure):
    num_rows = math.floor(math.sqrt(len(values)))
    num_cols = len(values) // num_rows
    num_cols = num_cols if len(values) % num_rows == 0 else num_cols + 1

    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[value[0] for value in values],
        horizontal_spacing=0.07,
    )

    for i, value in enumerate(values):
        row = i // num_cols + 1
        col = i % num_cols + 1
        fig.update_xaxes(
            title_text="hop number",
            row=row,
            col=col,
            tickfont_size=9,
            titlefont_size=10,
        )
        fig.update_yaxes(
            title_text="{} measure".format(measure),
            row=row,
            col=col,
            tickfont_size=9,
            titlefont_size=10,
            ticksuffix=" ",
        )
        fig.add_trace(
            go.Scatter(x=value[1], y=value[2], mode="lines", showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=900,
        width=1000,
        title_text="<b> Evolution of {} measure on shortest path \b ".format(
            str(measure)
        ),
        title_x=0.5,
    )
    fig.update_annotations(font_size=13)

    fig.show(renderer="svg")


# Compute similarities from each node to the target node for the shortest path of the top K most played games
def get_sim_shortest_path(k, G_articles, df_paths_finished, df_embeddings):
    top_k_games = get_top_k_games(df_paths_finished, k)
    names = []
    for i in top_k_games:
        names.append("{} to {}".format(i[0], i[1]))
    top_k_games_paths = []
    similarities = []
    for pair in set(top_k_games):
        top_k_games_paths.append(get_shortest_path(G_articles, pair[0], pair[1]))

    for i in range(k):
        list_of_articles = top_k_games_paths[i][:-1]
        target_article = top_k_games_paths[i][-1]
        similarities.append(
            compute_cosine_similarity(df_embeddings, list_of_articles, target_article)
        )
    y = [l.tolist() for l in similarities]
    y = [sum(s, []) for s in y]
    x = [list(range(len(y))) for y in y]

    return list(zip(names, x, y))


# We define a function to find how many different players have played certain games
def get_number_players(df_paths_finished, top_k_games):
    top_start = [i[0] for i in top_k_games]
    top_target = [i[1] for i in top_k_games]
    top_paths = list(zip(top_start, top_target))

    df_pruned = df_paths_finished.merge(
        pd.DataFrame(top_paths, columns=[DataSchema.START, DataSchema.TARGET]),
        how="inner",
        on=[DataSchema.START, DataSchema.TARGET],
    )
    return df_pruned.hashedIpAddress.nunique()
