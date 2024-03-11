from googleapiclient.discovery import build
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm  # to plot bell curve
from sklearn.metrics import f1_score, precision_score, recall_score # to calculate F1 score, precision and recall score

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# YouTube API credentials
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCQOAigDInp_oVaCY4QStVyj1AUSxMRgxE"

# Function to fetch comments from YouTube video
@st.cache_data
def fetch_youtube_comments(video_id):
    youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    comments = []

    # Fetch comments using pagination
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Maximum number of comments per page
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])

        # Check if there are more pages
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

# Function to clean text and count word occurrences
@st.cache_data
def clean_and_count(text):
    cleaned_text = cleantext.clean(text, clean_all=False, extra_spaces=True,
                                   stopwords=True, lowercase=True, numbers=True, punct=True)
    # Tokenize the cleaned text into words
    words = cleaned_text.split()

    # Count the occurrences of each word
    word_counts = Counter(words)

    # Create a DataFrame from the word counts
    word_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])

    # Sort the DataFrame by count in descending order
    word_df = word_df.sort_values(by='Count', ascending=False)

    return word_df

# Function to analyze sentiment and return sentiment label - for uploading function
@st.cache_data
def analyze(x, threshold):
    if x >= threshold:
        return 'Positive'
    elif x <= -threshold:
        return 'Negative'
    else:
        return 'Neutral'
    
# Function to analyze sentiment and return sentiment label - for YouTube video function
@st.cache_data
def analyze_yt(x):
    if x > 0:
        return 'Positive'
    elif x < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Preprocess comments to handle timestamps in YouTube video comments
def preprocess_comments(text):
    # Split the text by newline characters to handle multiple comments
    comments = text.split('\n')
    formatted_comments = []

    for comment in comments:
        # Split the comment by space to handle multiple words
        words = comment.split()
        formatted_comment = []

        for word in words:
            # Check if the word matches a timestamp pattern (e.g., 00:00)
            if ':' in word and all(char.isdigit() or char == ':' for char in word):
                # Format the timestamp as desired, for example, enclosing it in parentheses
                formatted_word = f"({word})"
            else:
                formatted_word = word

            formatted_comment.append(formatted_word)

        # Join the words back to form the formatted comment
        formatted_comments.append(' '.join(formatted_comment))

    # Join the formatted comments with newline characters
    return '\n'.join(formatted_comments)

# Page 1: Introduction & Background
def page_introduction():
    st.title('Automated Sentiment Analysis')
    st.write("""
    Understanding sentiment is crucial for deciphering the collective mood and emotions within various contexts.
    Our tool aims to streamline sentiment analysis processes, providing valuable insights into public opinion for diverse applications.

    **Purpose:**
    1. Gain insights into public sentiment for informed decision-making.
    2. Analyze sentiment trends to understand audience preferences and concerns.
    3. Facilitate effective marketing strategies by gauging brand perception.

    **How to Start:**
    1. Clean Text: Enter the text you want to analyze and clean it for sentiment analysis.
    *(i.e. The 10 cakes bought from XYZ Bakery were very delicious. I LOVE IT!)*
    2. Perform Sentiment Analysis: Input your text to receive sentiment polarity and subjectivity scores.
    """)
    pre = st.text_input('1. Clean Text: ')
    if pre:
         #st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 #stopwords=True ,lowercase=True ,numbers=True , punct=True))
        
    text = st.text_input('Perform Sentiment Analysis: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', blob.sentiment.polarity)
        st.write('Subjectivity: ', blob.sentiment.subjectivity)

    st.write("""
    Polarity: Polarity score measures the sentiment expressed in text, ranging from negative (-1) to positive (+1). It quantifies the degree of emotional tone conveyed by words or phrases in a text.
    
    Subjectivity: Subjectivity score indicates the degree to which a piece of text expresses opinions, emotions, or personal beliefs rather than factual information. It quantifies the subjectiveness of language, ranging from purely objective (0) to highly subjective/opinionated expressions (+1).
    """)

    st.write("""
    **Navigate to more tools on the left < < <**
    """)
        
            
# Page 2: Analyze Text In XLSX File
def page_analyze_xlsx():
    st.title('Analyze Text In XLSX File')
    st.write('''
        Upload your XLSX file and analyze sentiment through visualisation.
        
        **Disclaimer:** Before uploading your file, please ensure the column containing the text to analyze is labeled as **comments** in the header. Failure to adhere to these guidelines may result in unexpected behavior during analysis.
        ''')
    #- Ensure that the column containing dates is labeled as **date** and the column is in text or date format.
    
    upl = st.file_uploader('Upload File')


    if upl:
            df = pd.read_excel(upl, engine='openpyxl')
        
            # Handle missing values
            df['comments'].fillna('', inplace=True)

            # Check data types
            df['comments'] = df['comments'].astype(str)

            # Calculate sentiment score
            df['score'] = df['comments'].apply(lambda x: TextBlob(x).sentiment.polarity)

            # Calculate mean and standard deviation of sentiment scores
            mean_score = df['score'].mean()
            std_dev = df['score'].std()
    
            # Plot the bell curve (normal distribution curve)
            st.write("**Bell Curve for Sentiment Scores**")
            plt.figure(figsize=(8, 4))
            #plt.title('Bell Curve for Sentiment Scores')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')

            # Generate values for the x-axis
            x_axis = np.linspace(-1, 1, 100)  # Adjust the range of values for the x-axis

            # Plot the bell curve
            plt.plot(x_axis, norm.pdf(x_axis, mean_score, std_dev))
            st.pyplot(plt)

            # Print mean and standard deviation
            st.write(f"Mean Score: {round(mean_score, 4)}")
            st.write(f"Standard Deviation: {round(std_dev, 4)}")

            # Dynamic threshold adjustment
            st.write("**Threshold Adjustment**")
            st.write("This determines the boundary between positive, negative, and neutral sentiment.")
            suggested_threshold = std_dev / 2  # Example adjustment based on standard deviation
            threshold_adjusted = st.slider("Select Threshold", min_value=0.1, max_value=1.0, step=0.1, value=suggested_threshold)

            # Step 2: Analyze Sentiments with Adjusted Threshold
            df['analysis'] = df['score'].apply(lambda x: analyze(x, threshold_adjusted))

            # Step 3: Display Results
            st.write(df)
        
            # Label true and predicted columns to calculate F1 Score
            true_sentiment = df['analysis']
            predicted_sentiment = df['score'].apply(lambda x: 'Positive' if x >= suggested_threshold else 'Negative' if x <= -suggested_threshold else 'Neutral')

            # Calculate F1, precision and recall score
            f1 = f1_score(true_sentiment, predicted_sentiment, average='weighted')
            precision = precision_score(true_sentiment, predicted_sentiment, average='weighted')
            recall = recall_score(true_sentiment, predicted_sentiment, average='weighted')

            # Display F1 score
            st.write(f"Precision: {round(precision,4)}")
            st.write(f"Recall: {round(recall,4)}")
            st.write(f"F1 Score: {round(f1,4)}")


            st.write("**Overall Counts**")
            st.write(f"Total 💬: {len(df)}")
            st.write(f"😊: {len(df[df['analysis'] == 'Positive'])}")
            st.write(f"😞: {len(df[df['analysis'] == 'Negative'])}")
            st.write(f"😐: {len(df[df['analysis'] == 'Neutral'])}")

            # Step 4: Visualizations and Further Analysis
            # Compute percentage of positive, negative, and neutral comments
            sentiment_counts = df['analysis'].value_counts()
            total_comments = len(df)
            percentage_positive = (sentiment_counts.get('Positive', 0) / total_comments) * 100
            percentage_negative = (sentiment_counts.get('Negative', 0) / total_comments) * 100
            percentage_neutral = (sentiment_counts.get('Neutral', 0) / total_comments) * 100

            # PLOT 1: Create pie chart title
            st.write("**Overall Sentiment Distribution**")

            # Plot donut chart for sentiment distribution
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [percentage_positive, percentage_negative, percentage_neutral]
            colors = ['green', 'red', 'orange']
            fig, ax = plt.subplots(figsize=(5, 2))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                                      wedgeprops=dict(width=0.3, edgecolor='w'))

            plt.setp(autotexts, size=5, weight="bold")  # Set font size for percentage labels
            plt.setp(texts, size=5)  # Set font size for labels

            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            #ax.set_title("Sentiment Distribution")

            # Draw a circle at the center of pie to make it look like a donut
            centre_circle = plt.Circle((0, 0), 0.8, color='white', linewidth=0.3)
            fig.gca().add_artist(centre_circle)

            st.pyplot(fig)
    
            # Display top 10 most used words
            word_df = clean_and_count(' '.join(df['comments']))
            st.write("**Word Sentiment Analysis**")
            st.write("Top 10 Most Used Words")
            #st.write(word_df.head(10))

            # Initialize lists to store counts for each sentiment category for each word
            positive_counts = []
            negative_counts = []
            neutral_counts = []

            # Calculate sentiment distribution for each top 10 word
            for word in word_df['Word'][:10]:
                word_sentiments = df[df['comments'].str.contains(word)]['analysis'].value_counts()
                positive_count = word_sentiments.get('Positive', 0)
                negative_count = word_sentiments.get('Negative', 0)
                neutral_count = word_sentiments.get('Neutral', 0)
                positive_counts.append(positive_count)
                negative_counts.append(negative_count)
                neutral_counts.append(neutral_count)

            # PLOT 2: Plot bar plot for top 10 most used words with sentiment counts
            plt.figure(figsize=(12, 8))
            # Adjust positions for the bars
            bar_width = 0.3
            index = np.arange(len(word_df['Word'][:10]))

            plt.barh(index, positive_counts, bar_width, color='green', label='Positive')
            plt.barh(index + bar_width, negative_counts, bar_width, color='red', label='Negative')
            plt.barh(index + 2 * bar_width, neutral_counts, bar_width, color='orange', label='Neutral')

            plt.xlabel('Count')
            plt.ylabel('Words')
            #plt.title('Top 10 Most Used Words')
            plt.yticks(index + bar_width, word_df['Word'][:10])
            plt.legend()
            plt.tight_layout()
            st.pyplot(plt)

            #PLOT 3: Bar Plot of Neutral Words
            # Calculate sentiment score for each word in the DataFrame
            word_df['word_score'] = word_df['Word'].apply(lambda word: round(TextBlob(word).sentiment.polarity, 2))

            # Determine neutral words based on the threshold
            # Select the top 10 most used neutral words
            top_10_neutral_words = word_df[(word_df['Count'] > 0) & (word_df['word_score'] >= -suggested_threshold) & (word_df['word_score'] <= suggested_threshold)].head(10)

            # Create a horizontal bar chart for the top 10 neutral words with their counts
            st.write("Top 10 Most Used Neutral Words")
            plt.figure(figsize=(10, 6))
            plt.barh(top_10_neutral_words['Word'], top_10_neutral_words['Count'], color='orange')
            plt.xlabel('Count')
            plt.ylabel('Words')
            #plt.title('Top 10 Most Used Neutral Words')
            plt.gca().invert_yaxis()  # Invert y-axis to display words from top to bottom
            plt.show()
            st.pyplot(plt)

    st.write("""
    **Navigate to more tools on the left < < <**
    """)

        
#YOUTUBE VIDEO FUNCTION
# Page 3: Analyze YouTube Video Comments
def page_analyze_youtube():
    st.title('Analyze YouTube Video Comments')
    st.write("""
    Enter the YouTube Video ID to fetch comments and analyze sentiment through visualisation.
    """)

    video_id = st.text_input("https://www.youtube.com/watch?v= [Video ID]", value="",
                                 help="[Enter ID here. It is a string of code at the end of the link. i.e. D56_Cx36oGY]")

    if st.button('Fetch Comments'):
            if video_id:
                try:
                    # Fetch comments from YouTube API
                    comments = fetch_youtube_comments(video_id)

                    # Preprocess comments to handle timestamps
                    comments['text'] = comments['text'].apply(preprocess_comments)

                    # Create DataFrame from comments
                    df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])

                    # Handle missing values
                    df['text'].fillna('', inplace=True)
            
                    # Calculate sentiment score for each comment
                    #df['score'] = df['text'].apply(lambda x: round(TextBlob(x).sentiment.polarity, 4))
                    #df['score'] = df['text'].apply(lambda x: format(float(TextBlob(x).sentiment.polarity), '.4f'))
                    df['score'] = df['text'].apply(lambda x: float(TextBlob(x).sentiment.polarity))
                

                    # Calculate mean and standard deviation of sentiment scores
                    mean_score = df['score'].mean()
                    std_dev = df['score'].std()

                    # Step 2: Bell Curve with Mean, Standard Deviation, Dynamic Threshold
                    st.write("**Bell Curve for Sentiment Scores**")
                    plt.figure(figsize=(8, 4))
                    plt.xlabel('Sentiment Score')
                    plt.ylabel('Frequency')
                    x_axis = np.linspace(-1, 1, 100)  # Adjust the range of values for the x-axis
                    plt.plot(x_axis, norm.pdf(x_axis, mean_score, std_dev))
                    st.pyplot(plt)

                    # Print mean and standard deviation rounded to two decimal places
                    st.write(f"Mean Score: {round(mean_score, 4)}")
                    st.write(f"Standard Deviation: {round(std_dev, 4)}")

                    # Step 3: Display Results
                    df['text'] = df['text'].apply(preprocess_comments)  # Preprocess comments before displaying
                    df['analysis'] = df['score'].apply(lambda x: analyze_yt(x))
                
                    st.write("**Analysis Results**")
                    st.write(df)

                    st.write("**Overall Counts**")
                    st.write(f"Total 💬: {len(df)}")
                    st.write(f"😊: {len(df[df['analysis'] == 'Positive'])}")
                    st.write(f"😞: {len(df[df['analysis'] == 'Negative'])}")
                    st.write(f"😐: {len(df[df['analysis'] == 'Neutral'])}")

                    # Step 4: Visualizations and Further Analysis
                    sentiment_counts = df['analysis'].value_counts()
                    total_comments = len(df)
                    percentage_positive = (sentiment_counts.get('Positive', 0) / total_comments) * 100
                    percentage_negative = (sentiment_counts.get('Negative', 0) / total_comments) * 100
                    percentage_neutral = (sentiment_counts.get('Neutral', 0) / total_comments) * 100

                    #PLOT 1: Pie Chart Distribution
                    st.write("**Overall Sentiment Distribution**")

                    labels = ['Positive', 'Negative', 'Neutral']
                    sizes = [percentage_positive, percentage_negative, percentage_neutral]
                    colors = ['green', 'red', 'orange']
                    fig, ax = plt.subplots(figsize=(5, 2))
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                                              wedgeprops=dict(width=0.3, edgecolor='w'))

                    plt.setp(autotexts, size=5, weight="bold")
                    plt.setp(texts, size=5)

                    ax.axis('equal')  
                    centre_circle = plt.Circle((0, 0), 0.8, color='white', linewidth=0.3)
                    fig.gca().add_artist(centre_circle)

                    st.pyplot(fig)

                    #PLOT 2: Plot bar plot for top 10 most used words with sentiment counts
                    word_df = clean_and_count(' '.join(df['text']))
                    st.write("**Word Sentiment Analysis**")
                    st.write("Top 10 Most Used Words")
    
                    positive_counts = []
                    negative_counts = []
                    neutral_counts = []

                    for word in word_df['Word'][:10]:
                        word_sentiments = df[df['text'].str.contains(word)]['analysis'].value_counts()
                        positive_count = word_sentiments.get('Positive', 0)
                        negative_count = word_sentiments.get('Negative', 0)
                        neutral_count = word_sentiments.get('Neutral', 0)
                        positive_counts.append(positive_count)
                        negative_counts.append(negative_count)
                        neutral_counts.append(neutral_count)

                    plt.figure(figsize=(12, 8))
                    bar_width = 0.3
                    index = np.arange(len(word_df['Word'][:10]))

                    plt.barh(index, positive_counts, bar_width, color='green', label='Positive')
                    plt.barh(index + bar_width, negative_counts, bar_width, color='red', label='Negative')
                    plt.barh(index + 2 * bar_width, neutral_counts, bar_width, color='orange', label='Neutral')

                    plt.xlabel('Count')
                    plt.ylabel('Words')
                    plt.yticks(index + bar_width, word_df['Word'][:10])
                    plt.legend()
                    plt.tight_layout()
                    st.pyplot(plt)

                    #PLOT 3: Bar Plot of Neutral Words
                    word_df['word_score'] = word_df['Word'].apply(lambda word: round(TextBlob(word).sentiment.polarity, 4))

                    top_10_neutral_words = word_df[(word_df['Count'] > 0) & (word_df['word_score'] == 0)].head(10)

                    st.write("Top 10 Most Used Neutral Words")
                    plt.figure(figsize=(10, 6))
                    plt.barh(top_10_neutral_words['Word'], top_10_neutral_words['Count'], color='orange')
                    plt.xlabel('Count')
                    plt.ylabel('Words')
                    plt.gca().invert_yaxis()
                    plt.show()
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning('Please enter a valid YouTube Video ID')

    st.write("""
    **Navigate to more tools on the left < < <**
    """)
    
        

# Create a multipage layout using a dictionary
pages = {
    "Introduction & Background": page_introduction,
    "Analyze Text In XLSX File": page_analyze_xlsx,
    "Analyze YouTube Video Comments": page_analyze_youtube
}

# Create a sidebar navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Execute the selected page function
if selection in pages:
    pages[selection]()
