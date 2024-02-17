# The Untold Truth Behind Wikispeedia 
***Wikispeedia is a game based on Wikipedia, where players are given a start and end article and they must reach their goal in the fastest time possible only by using Wikipedia hyperlinks.***

***This study aims at analysing players strategies in order to leverage out insightful conclusions about their navigational behaviour.***

## Abstract

The idea behind this study takes its roots in an observation of the behaviour of Wikispeedia's players. After carefully studying some rounds of this game, we observed that most players would rely on the same strategy. At first, players tend to go towards articles that give them the most options for the next hop. This is probably the case to avoid getting trapped in a dead-end path. Once they feel closer to the target, they go into more specific topics that are closely related to the goal.

This behaviour also enables us to leverage insightful information about semantical distances between topics. As the paths are actually played by humans, there must be some logic to the hops chosen that should reflect that person's grasp on semantical closeness between the topics at hand.

In light of this, we articulate our story around three main chapters :

1. Can this behaviour be generalised?
2. What is the semantic meaning of these paths?
3. Is this strategy optimal?

We believe that, by analysing the three aforementioned axis, we will be able to provide a relevant insightful analysis of the observed behaviour.

## Research Questions

Through the different steps of our proposed analysis, we plan on investigating the following questions :

1. Is there a behaviour that can be generalized?

   - Are the most central nodes the most visited ones?
   - What is the evolution of the centrality of the nodes visited in the players' path?
   - What is the evolution of the centrality of the nodes visited in the shortest path?
2. What is the semantic meaning of the players' paths?
 
   - Is the player getting semantically closer to the goal?
   - Does shortest path reflect semantic closeness/similarity?
3. Are there any strategies to avoid?

   - How do people that do not succeed behave?
   - How do they differ from winners in their decision making?


## Website & Data Story 
The website that presents the data story can be accessed by following this [link](https://mehdizayene-adaes-wikispeedia-homepage-67asrh.streamlit.app/).
The website contains two main pages : the homepage and the data story page. You can seemlessly navigate between both pages using the index on the top left of the website.

### Homepage
The homepage presents the dataset and allows the reader to effectively play a game of Wikispeedia :
- By clicking on the *Start the game!* button, a path will be drawn for the player to achieve. 
- Displayed on the screen are the node count, the current and target articles as well as a *get hint* button if you ever feel stuck. This will find the shortest path to the target from the current node and help the player by giving the first node on that path.
- The choice can be made in the selection box by clicking on the article and then on the *submit* button. An *abandon* button is also there to allow the player to give up at any point. 
- Finally, a snippet of the current article is displayed on top to allow the user to make an informed decision about the next article. 

### Data Story
This page contains the narrative of our data story. 


## Methods

**Step 1 :**

- Data Loading
- Data Pre-Processing (values sanity checks / encode backtracks in dropped paths / removal of unplayed games)
- Derive helper methods in a separate file
- Creation of [networkX](https://networkx.org/) graph for centrality measure analysis and shortest paths
- Creation of DataFrames for document embeddings
- Exploratory Data Analysis

**Step 2 :**

- Investigation of different [centrality metrics](https://en.wikipedia.org/wiki/Centrality) to be used (out-degree / closeness / PageRank)
- Rank nodes according to best centrality metric
- Plot the evolution of the chosen metric by each hop of the paths
- Draw statistical conclusions : Is the proposed behaviour generalisable?

**Step 3 :**

- We define the distance metric as the shortest path distance from point to the end article (as can be found in the dataset)
- For each hop of the considered path, compute the distance from that hop to the goal using the defined metric
- Plot the evolution of the metric as we progress through the path

**Step 4 :**

- Use [sBERT](https://www.sbert.net/) to provide vectorial embeddings of articles
- Compute cosine similarity between embedded vectors to yield semantic closeness
- Investigate whether or not the shortest path reflects semantic closeness between consecutive hops

**Step 5 :**

- See if the players are getting semantically closer to the destination by each hop they take
- Plot the evolution of the semantic closeness from each hop to the target page as we progress through the path
- Observe and draw conclusion on players' behaviour

**Step 6 :**

- Study looser paths and investigate if they follow the same strategy
- Find out what differs in their decision making
- Wrap up and draw final conclusions about **predictability of human behaviour**
- Present everything in the form of story telling

## Proposed Timeline

- **Step 1** : 16/11/2022
- **Step 2** : 25/11/2022
- **Step 3 & 4** : 5/12/2022
- **Step 5** : 12/12/2022
- **Step 6** : 21/12/2022

## Organisation within the team

- **Step 1** : Acheche Khalil & Chaouch Yassine
- **Step 2** : Chaouch Yassine & Acheche Khalil
- **Step 3 & 4** : Mouaffak Selim & Zayene Mehdi
- **Step 5** : Zayene Mehdi & Mouaffak Selim
- **Step 6** : All Team

## How to use

### Content

This repository contains three main files :

- **The_Untold_Truth_Behind_Wikispeedia.ipynb** : this notebook contains the main storytelling steps of our analysis and the entirety of the data story
- **Data_Engineering.ipynb** : This file contains the major preprocessing steps as well as the construction of useful data structures.
- **helpers.py** : helper functions and tool methods.


### Data 
- All the data needed for smooth execution of the notebook can be found in the **data** folder (including the article embeddings).

### Installation

- Make sure to install all required packages by running the following command :
  ``$ pip install -r requirements.txt ``

## Credits

This project was realised by four EPFL (École Polytechnique Fédérale de Lausanne) students :

* Zayene Mehdi
* Chaouch Yassine
* Acheche Khalil
* Mouaffak Selim

We would like to thank Prof. West and his team for their guidance throughout this project.
