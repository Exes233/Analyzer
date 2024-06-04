from django.shortcuts import render, redirect
from . forms import CreateUserForm, LoginForm, EditProfileForm
from django.contrib.auth.models import auth
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import HashtagMetrics, HashtagMetricsX
import praw
from collections import defaultdict
from rake_nltk import Rake
from datetime import datetime
import re
import math
import nltk
from nltk.tokenize import word_tokenize
import requests
from PIL import Image
import pytesseract
from io import BytesIO
import imghdr
import cv2
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from django.http import JsonResponse


def collect_hashtag_metrics(subr):
        print(f'Collecting posts for r/{subr}:')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def lemmatize_text(text):
            tokens = word_tokenize(text)
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)

        def extract_hashtags(text):
            tokenizer = RegexpTokenizer(r'\b\w[\w-]*\b')
            words = tokenizer.tokenize(text)
            hashtags = [word[1:] for word in words if word.startswith('#')]
            return hashtags

        def generate_hashtags(text):
            if not text.strip() or all(word.lower() in stop_words for word in tokenizer.tokenize(text)):
                return []

            text = re.sub(r'[^a-zA-Z\s]', '', text)

            lemmatized_text = lemmatize_text(text)

            vectorizer = TfidfVectorizer(max_features=25, ngram_range=(1, 3))

            if lemmatized_text.strip():
                tfidf_matrix = vectorizer.fit_transform([lemmatized_text])

                feature_names = vectorizer.get_feature_names_out()


                top_keywords = sorted(zip(tfidf_matrix.toarray()[0], feature_names), reverse=True)[:5]

                hashtags = ['#' + re.sub(r'(to|for|and|from|in|on|at|with|by|about|of|as|into|like|through|between|after|before|under|over|without|around|among)$', '', re.sub(r'^(to|for|from|in|on|at|with|by|about|of|as|into|like|through|between|after|before|under|over|without|around|among|and)', '', keyword, flags=re.IGNORECASE), flags=re.IGNORECASE).strip().replace(' ', '_')
                    for _, keyword in top_keywords
                    if not keyword.isdigit() and len(keyword) > 4
                    and not any(pronoun in keyword.lower() for pronoun in ["i", "you", "he", "she", "it", "we", "they"])
                    and keyword.lower() not in ["what", "that", "why"] ]
            else:
                hashtags = []

            return hashtags


        def extract_text_from_image(image_url):
            try:
                response = requests.get(image_url)
                if response.ok:
                    image_bytes = response.content
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
                    text = pytesseract.image_to_string(binary_image, timeout=5)
                    return text
                else:
                    print("Failed to load the image.")
                    return ""
            except Exception as e:
                print(f"Error processing the image: {e}")
                return ""

        def is_image(url):
            try:
                response = requests.head(url, timeout=2)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'image' in content_type:
                        return True
                    image_type = imghdr.what(None, response.content)
                    return image_type is not None
            except requests.Timeout:
                print("Время запроса истекло. Продолжаем работу приложения.")
            except requests.RequestException as e:
                print(f"Произошла ошибка при выполнении запроса: {e}")
        
        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

        def merge_similar_hashtags(hashtags):
            merged_hashtags = defaultdict(list)
    
            # Группируем хештеги по их схожести
            for hashtag in hashtags:
                merged = False
                for merged_hashtag in merged_hashtags:
                    if similar(hashtag.lower(), merged_hashtag.lower()) > 0.6:
                        merged_hashtags[merged_hashtag].append(hashtag)
                        merged = True
                        break
                if not merged:
                    merged_hashtags[hashtag] = [hashtag]
    
          
            final_hashtags = []
            for merged_hashtag, similar_hashtags in merged_hashtags.items():
                shortest_hashtag = min(similar_hashtags, key=len)
                final_hashtags.append(shortest_hashtag)
    
            return final_hashtags
        def group_similar_tags(tags):
            grouped_tags = {}
    
            
            for tag in tags:
                matched = False
              
                for group_tag, group in grouped_tags.items():
                    if fuzz.ratio(group_tag, tag) > 60:  
                       
                        group.append(tag)
                        matched = True
                        break
                if not matched:
                    # Создаем новую группу для тега
                    grouped_tags[tag] = [tag]

            
            shortest_tag_groups = {}
            for group_tag, group in grouped_tags.items():
                shortest_tag = min(group, key=len)
                shortest_tag_groups[shortest_tag] = group

            return shortest_tag_groups
    
        reddit = praw.Reddit(
            client_id="1IY1RHSRp8tYAI3c53zvtA",
            client_secret="lKqt01vPDQCeAD8kVjiy0XyrG1l05g",
            user_agent="NURE_Project by	u/Sufficient_Initial40"
        )
        subreddit_name = subr

        subreddit = reddit.subreddit(subreddit_name)
        posts = subreddit.new(limit=100)

        hashtags_metrics = defaultdict(lambda: {'mentions': 0, 'votes': 0, 'comments': 0, 'age_hours': 0})
        index = 1
        for post in posts:
            print(f'Processing Reddit Post {index} / {100}')
            index += 1
            post_text = post.selftext
            post_hashtags = extract_hashtags(post_text)
            all_hashtags = post_hashtags

            if post.title.strip():
                title_hashtags = generate_hashtags(post.title)
                generated_hashtags = generate_hashtags(post_text) + title_hashtags
                all_hashtags += generated_hashtags

            if post.media and 'type' in post.media and post.media['type'] == 'image':
                image_url = post.media['url']
                image_text = extract_text_from_image(image_url)
                image_hashtags = generate_hashtags(image_text)
                all_hashtags += image_hashtags

            if 'http' in post_text:
                urls = re.findall(r'(https?://\S+)', post_text)
                for url in urls:
                    if is_image(url):
                        media_hashtags = generate_hashtags(extract_text_from_image(url))
                        all_hashtags += media_hashtags
            if is_image(post.url):
                image_text = extract_text_from_image(post.url)
                if image_text.strip():
                    url_hashtags = generate_hashtags(image_text)
                    all_hashtags += url_hashtags
            
            all_hashtags = merge_similar_hashtags(all_hashtags)
            post_time = datetime.utcfromtimestamp(post.created_utc)
            post_time_rounded = post_time.replace(minute=0, second=0, microsecond=0)
            for hashtag in all_hashtags:
                hashtags_metrics[hashtag]['mentions'] += 1
                hashtags_metrics[hashtag]['votes'] += post.score
                hashtags_metrics[hashtag]['comments'] += post.num_comments
                post_age = (datetime.utcnow() - post_time).total_seconds() / 3600
                hashtags_metrics[hashtag]['age_hours'] += post_age
            print("\n")
        sorted_hashtags_metrics = sorted(hashtags_metrics.items(), key=lambda x: x[1]['votes'], reverse=True)
        print('All Reddit posts processed, analysing now')
        dataset = [{'Tag': hashtag,
                'Mentions': metrics['mentions'],
                'Upvotes': metrics['votes'],
                'Comments': metrics['comments'],
                'First_Post_Time_Hours_Ago': math.floor(metrics['age_hours'])}
                for hashtag, metrics in sorted_hashtags_metrics
                if metrics['votes'] >= 0]
        df = pd.DataFrame(dataset)

        tags = [item['Tag'] for item in dataset]
  
        tag_groups = group_similar_tags(tags)
    
        combined_metrics = {}
        for group_tag, tags in tag_groups.items():
            group_metrics = {
                'Mentions': sum(item['Mentions'] for item in dataset if item['Tag'] in tags),
                'Upvotes': sum(item['Upvotes'] for item in dataset if item['Tag'] in tags),
                'Comments': sum(item['Comments'] for item in dataset if item['Tag'] in tags),
                'First_Post_Time_Hours_Ago': max(item['First_Post_Time_Hours_Ago'] for item in dataset if item['Tag'] in tags)
            }
            combined_metrics[group_tag] = group_metrics
       
        combined_dataset = [{'Tag': tag,
                    'Mentions': metrics['Mentions'],
                    'Upvotes': metrics['Upvotes'],
                    'Comments': metrics['Comments'],
                    'First_Post_Time_Hours_Ago': metrics['First_Post_Time_Hours_Ago']}
                    for tag, metrics in combined_metrics.items()]

     
        combined_df = pd.DataFrame(combined_dataset)
        df = combined_df
        weight_mentions = 0.25
        weight_upvotes = 0.25
        weight_comments = 0.25
        weight_time = 0.25

        df['Normalized_Mentions'] = 1 / (1 + df['Mentions'])
        df['Normalized_Upvotes'] = df['Upvotes'] / df['Upvotes'].max()
        df['Normalized_Comments'] = df['Comments'] / df['Comments'].max()

        max_time = df['First_Post_Time_Hours_Ago'].max()
        df['Normalized_Time'] = 1 - (df['First_Post_Time_Hours_Ago'] / max_time)
        df['Relative_Attraction'] = (
        weight_mentions * df['Normalized_Mentions'] +
        weight_upvotes * df['Normalized_Upvotes'] +
        weight_comments * df['Normalized_Comments'] +
        weight_time * df['Normalized_Time']
        )

        df['Relative_Attraction'] = df['Relative_Attraction'] / df['Relative_Attraction'].max()

        df_sorted = df.sort_values(by='Relative_Attraction', ascending=False)
        df_sorted.reset_index(drop=True, inplace=True)
        HashtagMetrics.objects.filter(topic = subr).delete()
        
        for index, row in df_sorted.iterrows():
            HashtagMetrics.objects.create(
                tag=row['Tag'],
                mentions=row['Mentions'],
                upvotes=row['Upvotes'],
                comments=row['Comments'],
                first_post_time_hours_ago=math.floor(row['First_Post_Time_Hours_Ago']),
                relative_attraction = round(row['Relative_Attraction'],2),
                topic = subreddit_name ,
                network = "Reddit"
            )

def process_tweets(search):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def lemmatize_text(text):
            tokens = word_tokenize(text)
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            return ' '.join(lemmatized_tokens)

    def extract_hashtags(text):
            hashtags = re.findall(r'#\w+', text)
            return hashtags

    def generate_hashtags(text):

        text = re.sub(r'[^a-zA-Z\s]', '', text)
        lemmatized_text = lemmatize_text(text)
        if not lemmatized_text.strip() or all(word.lower() in stop_words for word in tokenizer.tokenize(lemmatized_text)):
            return []

        vectorizer = TfidfVectorizer(max_features=25, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform([lemmatized_text])
        feature_names = vectorizer.get_feature_names_out()
        top_keywords = sorted(zip(tfidf_matrix.toarray()[0], feature_names), reverse=True)[:5]
        hashtags = ['#' + re.sub(r'(to|for|and|from|in|on|at|with|by|about|of|as|into|like|through|between|after|before|under|over|without|around|among)$', '', re.sub(r'^(to|for|from|in|on|at|with|by|about|of|as|into|like|through|between|after|before|under|over|without|around|among|and)', '', keyword, flags=re.IGNORECASE), flags=re.IGNORECASE).strip().replace(' ', '_')
                for _, keyword in top_keywords
                if not keyword.isdigit() and len(keyword) > 4
                and not any(pronoun in keyword.lower() for pronoun in ["i", "you", "he", "she", "it", "we", "they"])
                and keyword.lower() not in ["what", "that", "why"]]
        return hashtags



    def extract_text_from_image(image_url):
            try:
                response = requests.get(image_url)
                if response.ok:
                    image_bytes = response.content
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
                    text = pytesseract.image_to_string(binary_image, timeout=5)
                    return text
                else:
                    print("Failed to load the image.")
                    return ""
            except Exception as e:
                print(f"Error processing the image: {e}")
                return ""

    def is_image(url):
            try:
                response = requests.head(url, timeout=2)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type')
                    if content_type and 'image' in content_type:
                        return True
                    image_type = imghdr.what(None, response.content)
                    return image_type is not None
            except requests.Timeout:
                print("Время запроса истекло. Продолжаем работу приложения.")
            except requests.RequestException as e:
                print(f"Произошла ошибка при выполнении запроса: {e}")

    def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()

    def merge_similar_hashtags(hashtags):
            merged_hashtags = defaultdict(list)

            
            for hashtag in hashtags:
                merged = False
                for merged_hashtag in merged_hashtags:
                    if similar(hashtag.lower(), merged_hashtag.lower()) > 0.6:
                        merged_hashtags[merged_hashtag].append(hashtag)
                        merged = True
                        break
                if not merged:
                    merged_hashtags[hashtag] = [hashtag]

      
            final_hashtags = []
            for merged_hashtag, similar_hashtags in merged_hashtags.items():
                shortest_hashtag = min(similar_hashtags, key=len)
                final_hashtags.append(shortest_hashtag)

            return final_hashtags
    def group_similar_tags(tags):
            grouped_tags = {}

            
            for tag in tags:
                matched = False
                
                for group_tag, group in grouped_tags.items():
                    if fuzz.ratio(group_tag, tag) > 60:  
                        
                        group.append(tag)
                        matched = True
                        break
                if not matched:
                    
                    grouped_tags[tag] = [tag]

            
            shortest_tag_groups = {}
            for group_tag, group in grouped_tags.items():
                shortest_tag = min(group, key=len)
                shortest_tag_groups[shortest_tag] = group

            return shortest_tag_groups

    url = "https://twitter-api45.p.rapidapi.com/search.php"

    headers = {
	"X-RapidAPI-Key": "44a18acdfcmsh39c0f902bf49df9p1905d8jsn0f18c2a45b24",
	"X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
    }

    results = []

    querystring = {"query":search}
    response = requests.get(url, headers=headers, params=querystring)
    results.append(response)

    querystring = {"query":search,"cursor":response.json().get('next_cursor')}
    response = requests.get(url, headers=headers, params=querystring)
    results.append(response)

    querystring = {"query":search,"cursor":response.json().get('next_cursor')}
    response = requests.get(url, headers=headers, params=querystring)
    results.append(response)

    querystring = {"query":search,"cursor":response.json().get('next_cursor')}
    response = requests.get(url, headers=headers, params=querystring)
    results.append(response)

    hashtags_metrics_x = defaultdict(lambda: {'mentions': 0, 'retweets': 0, 'quotes': 0, 'views': 0, 'age_hours': 0})
    for response in results:
        if response.status_code == 200:
            search_results = response.json()
            tweets = search_results.get('timeline', [])
            if tweets:
                for tweet in tweets:
                    print("Getting tweet")
                    tweet_text = tweet.get('text')
                    tweet_hashtags = extract_hashtags(tweet_text)
                    all_hashtags = tweet_hashtags
                    all_hashtags = merge_similar_hashtags(all_hashtags)
                    tweet_time = datetime.strptime(tweet.get('created_at'), "%a %b %d %H:%M:%S %z %Y")
                    for hashtag in all_hashtags:
                        hashtags_metrics_x[hashtag]['mentions'] += 1
                        hashtags_metrics_x[hashtag]['retweets'] += int(tweet.get('retweets'))
                        hashtags_metrics_x[hashtag]['quotes'] += int(tweet.get('quotes'))
                        if tweet.get('views') is not None:
                            hashtags_metrics_x[hashtag]['views'] += int(tweet.get('views'))
                        post_age = (datetime.utcnow() - tweet_time.replace(tzinfo=None)).total_seconds() / 3600
                        hashtags_metrics_x[hashtag]['age_hours'] += post_age

            else:
                print("Результатов не найдено.")
        else:
            print(f'Ошибка: {response.status_code}, сообщение: {response.text}')
    sorted_hashtags_metrics_x = sorted(hashtags_metrics_x.items(), key=lambda x: x[1]['views'], reverse=True)
    dataset = [{'Tag': hashtag,
                'Mentions': metrics['mentions'],
                'Retweets': metrics['retweets'],
                'Quotes': metrics['quotes'],
                'Views': metrics['views'],
                'First_Post_Time_Hours_Ago': math.floor(metrics['age_hours'])}
                for hashtag, metrics in sorted_hashtags_metrics_x
                if metrics['views'] >= 0 and len(hashtag) > 1 ]
    df = pd.DataFrame(dataset)
    weight_mentions = 0.2
    weight_retweets = 0.2
    weight_quotes = 0.2
    weight_views = 0.3
    weight_time = 0.1

    df['Normalized_Mentions'] = 1 / (1 + df['Mentions'])
    df['Normalized_Retweets'] = df['Retweets'] / df['Retweets'].max()
    df['Normalized_Quotes'] = df['Quotes'] / df['Quotes'].max()
    df['Normalized_Views'] = df['Views'] / df['Views'].max()

    max_time = df['First_Post_Time_Hours_Ago'].max()
    df['Normalized_Time'] = 1 - (df['First_Post_Time_Hours_Ago'] / max_time)
    df['Relative_Attraction'] = (
        weight_mentions * df['Normalized_Mentions'] +
        weight_retweets * df['Normalized_Retweets'] +
        weight_quotes * df['Normalized_Quotes'] +
        weight_views * df['Normalized_Views'] +
        weight_time * df['Normalized_Time']
        )

    df['Relative_Attraction'] = df['Relative_Attraction'] / df['Relative_Attraction'].max()

    df_sorted = df.sort_values(by='Relative_Attraction', ascending=False)
    df_sorted.reset_index(drop=True, inplace=True)

    HashtagMetricsX.objects.filter(topic = search).delete()
    for index, row in df_sorted.iterrows():
            HashtagMetricsX.objects.create(
                tag=row['Tag'],
                mentions=row['Mentions'],
                retweets=row['Retweets'],
                quotes=row['Quotes'],
                views=row['Views'],
                first_post_time_hours_ago=math.floor(row['First_Post_Time_Hours_Ago']),
                relative_attraction = round(row['Relative_Attraction'],2),
                topic = search ,
                network = "Twitter"
            )
    print("Added new tweets to table")

def homepage(request):

    return render(request,'tagsapp/index.html')

def register(request):

    if request.user.is_authenticated:
        return redirect('')

    form = CreateUserForm()

    if request.method == "POST":

        form = CreateUserForm(request.POST)

        if form.is_valid():

            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username} succescfully!')
            return redirect("login")
        
    context = {'registerform': form}    
    return render(request,'tagsapp/register.html', context = context)

def login(request):

    if request.user.is_authenticated:
        return redirect('')

    form = LoginForm()

    if request.method == 'POST':

        form = LoginForm(request, data = request.POST)

        if form.is_valid():

            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password = password)

            if user is not None:

                auth.login(request,user)
                return redirect("dashboard")
    
    context = {'loginform':form}

    return render(request,'tagsapp/login.html', context=context)

@login_required(login_url="login")
def dashboard(request):

    hashtag_metrics = HashtagMetrics.objects.all()

    return render(request,'tagsapp/dashboard.html', {'hashtag_metrics': hashtag_metrics})

@login_required(login_url="login")
def profile(request):
    
    return render(request, 'tagsapp/profile.html')

@login_required(login_url="login")
def edit_profile(request):
    if request.method == 'POST':
        form = EditProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('profile')  
    else:
        form = EditProfileForm(instance=request.user)
    return render(request, 'tagsapp/edit-profile.html', {'form': form})

@login_required(login_url="login")
def update_dashboard(request):
    if request.method == 'POST':
        collect_hashtag_metrics('Python')
        collect_hashtag_metrics('politics')
        collect_hashtag_metrics('news')
        collect_hashtag_metrics('programming')
        collect_hashtag_metrics('sports')
        collect_hashtag_metrics('science')
        collect_hashtag_metrics('cars')
        collect_hashtag_metrics('memes')
        process_tweets('politics')
        process_tweets('Python')
        process_tweets('news')
        process_tweets('programming')
        process_tweets('sports')
        process_tweets('science')
        process_tweets('cars')
        process_tweets('memes')
        hashtag_metrics = HashtagMetrics.objects.all()
        return render(request, 'tagsapp/dashboard.html', {'hashtag_metrics': hashtag_metrics})
    else:
        return redirect('dashboard')

def generate_chart(request):
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        network = request.POST.get('social_network')
        content_category = request.POST.get('content_category')
        metric_type = request.POST.get('metric_type')
        tag_count = int(request.POST.get('tag_count'))
        if network == "reddit":
            tags = HashtagMetrics.objects.filter(topic=content_category).order_by('-' + metric_type)[:tag_count]
        else:
            tags = HashtagMetricsX.objects.filter(topic=content_category).order_by('-' + metric_type)[:tag_count]

        labels = []
        data = []

        for tag in tags:
            labels.append(tag.tag)  
            data.append(getattr(tag, metric_type))
        
        metric_type_str = ""
        match metric_type:
            case "relative_attraction":
                metric_type_str = "Relevance"
            case "comments":
                metric_type_str= "Comments"
            case "upvotes":
                metric_type_str= "Upvotes"
            case "mentions":
                metric_type_str= "Mentions"
            case "retweets":
                metric_type_str= "Retweets"
            case "quotes":
                metric_type_str= "Quotes"

        chart_data = {
            'labels': labels,  
            'data': data,      
            'type': 'bar',      
            'metric_type': metric_type_str
        }
        print(data)
        print(labels)
        return JsonResponse(chart_data)
    else:
        return JsonResponse({'error': 'Invalid request'})

def user_logout(request):

    auth.logout(request)

    return redirect("")
