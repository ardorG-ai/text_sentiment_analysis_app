# Automated Text Sentiment Analysis Webapp

Webapp Link: https://textsentimentanalysisapp.streamlit.app/
<br>
Demo Video: https://youtu.be/SpA4Y_5F2uQ

## Introduction & Background
Sentiment Analysis plays a crucial role in understanding the collective mood and emotions of the general public, providing valuable insights into various contexts. It involves analysing data and categorizing it based on the sentiments expressed within, thereby shedding light on people's preferences, desires, and concerns.

For marketing teams, understanding public sentiment surrounding their own brand or competitor brands is essential to create effective marketing strategies. It gives brands an idea of how well their products are perceived by their audience. The analysis maps out the boundary between the different sentiments, guiding decision-making processes and campaign directions. (Fairlie, 2024)

In the realm of influencer marketing, quick assessments are necessary to ensure alignment with target audiences and brand tone. Understanding the discourse surrounding influencers and the language used to describe them is crucial. Waiting for reports from social agencies or data team can be time-consuming, hindering timely decision-making.

Furthermore, social listening tools have limitations, being restricted to selected platforms like only Facebook or Twitter. Moreover, these tools may not accurately capture sentiment dynamics across different geographical markets. According to Meltwater’s Digital 2024 Global Overview Report (chart below), social media adoption varies significantly between countries. For instance, while Singapore averages 6.9 social platforms per user, the Philippines averages 8.0. This discrepancy underscores the challenge of finding comprehensive tools that cater to diverse market landscapes.

<img width="699" alt="Picture 1" src="https://github.com/Kfkyyian1/text_sentiment_analysis_app/assets/146427900/098fa442-757a-4df9-a306-e064c99db320"> <br>
Figure 1
(2024 Global Digital Report, 2024, pp. 233)
<p>
Not forgetting, sometimes data teams are overwhelmed, lacking the bandwidth needed to curate reports swiftly for timely decision-making. Hence, the objective of this project is to develop a tool that empowers users to analyse sentiment across various platforms more efficiently, provided they have access to the requisite raw data. By offering a versatile webapp, brands can gain comprehensive insights that accurately reflect the nuances of their target markets.</p>

## Tools Used
1.	Upload Feature: 
The upload feature serves as the primary avenue for users to upload raw data and have it analysed. This functionality is crucial as it allows users to upload data collected from various platforms, enabling a diverse range of data sources to be analysed seamlessly. The integration of the upload feature is facilitated by utilizing the pandas library, which enables the efficient reading of Excel (.xlsx) files.

2.	TextBlob Library: 
The TextBlob Python library is utilized for sentiment analysis due to its robustness and ease of implementation. It offers a ready-built library for processing textual data and provides a consistent API for common natural language processing (NLP) tasks such as sentiment analysis. By leveraging TextBlob, the sentiment polarity of each comment can be calculated efficiently. (Great Learning, 2022)

3.	Cleantext Module: 
The Cleantext module is employed to clean the text data. It facilitates various text preprocessing tasks such as removing extra spaces, stopwords, lowercase conversion, and punctuation removal. Additionally, Cleantext tokenizes the cleaned text into words and generates a DataFrame containing word counts, which forms the basis for further analysis.

4.	Collections Counter: 
The Collections Counter is utilized to count the occurrences of each word within the text data. This functionality enables the creation of a DataFrame that organizes words based on their frequency of occurrence, providing insights into the most commonly used words within the dataset.

5.	Streamlit: 
Streamlit is chosen as the framework for developing the web application due to its simplicity, flexibility, and real-time preview capabilities. Streamlit allows for the rapid creation of interactive dashboards directly from Python scripts, enabling seamless integration with data analysis workflows. Its intuitive interface streamlines the development process and allows preview and test the web application in real-time, saving valuable development time and effort.

6.	Slider for Threshold Adjustment: 
The inclusion of a slider for threshold adjustment provides users with greater flexibility and control over the sentiment analysis process. By allowing users to adjust the threshold dynamically, they can fine-tune the categorization of sentiments as positive, negative, or neutral, thereby enhancing the accuracy and relevance of the analysis results.

7.	F1 Score Calculation: 
The F1 score is calculated using the sklearn.metrics module to evaluate the performance of the sentiment analysis model. The F1 score provides a comprehensive measure of the model's precision and recall, balancing both metrics to assess overall performance accurately. By incorporating the F1 score calculation, users can gauge the effectiveness of the sentiment analysis model and make informed decisions based on its performance metrics.

8.	Visualization Tools (Matplotlib and scipy.stats): 
Matplotlib and scipy.stats are utilized for data visualization, enabling the creation of insightful visualizations such as bell curves, donut charts and horizontal bar charts. These visualizations offer a clear and intuitive representation of sentiment distribution and word sentiment analysis, allowing users to interpret the analysis results effectively and derive actionable insights from the data.

9. Integration with YouTube: 
The decision to integrate YouTube into the web application is supported by insights from Meltwater’s Digital 2024 Global Overview Report, which highlights this platform as one of the most widely used social platforms globally. Integrating YouTube functionality expands the reach of the sentiment analysis tool, enabling users to analyse sentiment across diverse content. Facebook was not selected due to the restriction of only moderators/admins of a page is allowed to fetch comments, as stated in their [GraphAPI documentation](https://developers.facebook.com/docs/graph-api/reference/page-post/comments/) .

<img width="696" alt="Picture 2" src="https://github.com/Kfkyyian1/text_sentiment_analysis_app/assets/146427900/52b0147a-a9b9-4000-903a-025189b1b82e"> <br>
Figure 3
(2024 Global Digital Report, 2024, pp. 232)
<p>
The selection of these tools and technologies is driven by their ability to facilitate efficient data analysis, provide intuitive user experiences, and enable seamless integration with external platforms, ultimately contributing to the development of the sentiment analysis web application.
</p>


## Proposed Methodology
The proposed methodology focuses on leveraging the automation system to enhance efficiency, decision-making capabilities, and brand perception.

### Time Savings and Simplified Reporting
Marketing teams and brands can save valuable time on the manual compilation and creation of sentiment analysis reports. Instead of manually building reports from scratch, users can effortlessly upload data files or input links to relevant content, allowing the system to generate comprehensive visualizations and insights automatically.

### Quick Decision-Making and Overview
Users can gain access to real-time sentiment analysis and visualizations, providing them with quick overviews of trends and patterns. This capability enables marketing teams and brands to make informed decisions promptly, responding swiftly to emerging trends, consumer feedback, and market dynamics.

### Customized Sentiment Analysis
Different products or brand categories may exhibit varying sentiment distributions within customer reviews. The automation system allows for the customization of sentiment analysis by adjusting the threshold parameters. For instance, luxury brands may opt for a higher threshold for positive sentiment compared to budget brands, reflecting the distinct expectations and perceptions associated with each product category.

In conclusion, the proposed methodology harnesses the capabilities of the automation system to revolutionize sentiment analysis practices, empowering marketing teams and brands with actionable insights, streamlined processes, and enhanced decision-making capabilities. By leveraging the webapp, organizations can gain a competitive edge and foster positive brand experiences in today's dynamic business landscape.

## References
Fairlie, M. (2024, January 17). How sentiment analysis can Improve your sales. 
Business News Daily. https://www.businessnewsdaily.com/10018-sentiment-analysis-improve-business.html

Great Learning. (2022, February 21). Textblob | NLP tutorial for Beginners | Natural 
Language Processing | Great Learning [Video]. YouTube. 
https://www.youtube.com/watch?v=_heNNAhSYx0

Vidito. (2023). GitHub - Vidito/textblob_sentiment_analysis: A streamlit python web 
app to analyze sentiment in a CSV file and add the sentiment values to the 
file. GitHub. https://github.com/Vidito/textblob_sentiment_analysis?tab=readme-ov-file

Global Digital Report. (2024). Meltwater. pp. 232 & 233. https://www.meltwater.com/en/global-digital-trends


