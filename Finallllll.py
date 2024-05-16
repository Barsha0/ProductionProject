import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import re


# Load the dataset
df = pd.read_csv('C:\\Users\\user\\OneDrive\\Desktop\\Production\\Dataset\\Dataset.csv')

# Assuming 'Type' is the target variable
X = df.drop(columns=['Type'])
y = df['Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)


def extract_features(url):
    # URL length
    url_length = len(url)

    # Number of dots
    number_of_dots_in_url = url.count('.')

    # Whether URL has repeated digits
    having_repeated_digits_in_url = bool(re.search(r'(\d)\1', url))

    # Number of digits in URL
    number_of_digits_in_url = len(re.findall(r'\d', url))

    # Number of special characters in URL
    number_of_special_char_in_url = len(re.findall(r'[^\w\s]', url))

    # Number of hyphens in URL
    number_of_hyphens_in_url = url.count('-')

    # Number of underscore in URL
    number_of_underline_in_url = url.count('_')

    # Number of slash in URL
    number_of_slash_in_url = url.count('/')

    # Number of question mark in URL
    number_of_questionmark_in_url = url.count('?')

    # Number of equal in URL
    number_of_equal_in_url = url.count('=')

    # Number of at in URL
    number_of_at_in_url = url.count('@')

    # Number of dollar in URL
    number_of_dollar_in_url = url.count('$')

    # Number of exclamation in URL
    number_of_exclamation_in_url = url.count('!')

    # Number of hashtag in URL
    number_of_hashtag_in_url = url.count('#')

    # Number of percent in URL
    number_of_percent_in_url = url.count('%')

    # Domain length
    domain_length = len(url.split('/')[2])

    # Number of dots in domain
    number_of_dots_in_domain = url.split('/')[2].count('.')

    # Number of hyphens in domain
    number_of_hyphens_in_domain = url.split('/')[2].count('-')

    # Having special characters in domain
    having_special_characters_in_domain = bool(re.search(r'[^\w\s]', url.split('/')[2]))

    # Number of special characters in domain
    number_of_special_characters_in_domain = len(re.findall(r'[^\w\s]', url.split('/')[2]))

    # Having digits in domain
    having_digits_in_domain = bool(re.search(r'\d', url.split('/')[2]))

    # Number of digits in domain
    number_of_digits_in_domain = len(re.findall(r'\d', url.split('/')[2]))

    # Having repeated digits in domain
    having_repeated_digits_in_domain = bool(re.search(r'(\d)\1', url.split('/')[2]))

    # Number of subdomains
    number_of_subdomains = len(url.split('/')[2].split('.')) - 2

    # Having dot in subdomain
    having_dot_in_subdomain = '.' in url.split('/')[2]

    # Having hyphen in subdomain
    having_hyphen_in_subdomain = '-' in url.split('/')[2]

    # Average subdomain length
    subdomains = url.split('/')[2].split('.')
    if len(subdomains) > 1:
        average_subdomain_length = sum(len(sub) for sub in subdomains) / len(subdomains)
    else:
        average_subdomain_length = 0

    # Average number of dots in subdomain
    average_number_of_dots_in_subdomain = url.split('/')[2].count('.') / number_of_subdomains if number_of_subdomains > 0 else 0

    # Average number of hyphens in subdomain
    average_number_of_hyphens_in_subdomain = url.split('/')[2].count('-') / number_of_subdomains if number_of_subdomains > 0 else 0

    # Having special characters in subdomain
    having_special_characters_in_subdomain = bool(re.search(r'[^\w\s]', url.split('/')[2]))

    # Number of special characters in subdomain
    number_of_special_characters_in_subdomain = len(re.findall(r'[^\w\s]', url.split('/')[2]))

    # Having digits in subdomain
    having_digits_in_subdomain = bool(re.search(r'\d', url.split('/')[2]))

    # Number of digits in subdomain
    number_of_digits_in_subdomain = len(re.findall(r'\d', url.split('/')[2]))

    # Having repeated digits in subdomain
    having_repeated_digits_in_subdomain = bool(re.search(r'(\d)\1', url.split('/')[2]))

    # Having path
    having_path = bool(url.split('/')[3:])

    # Path length
    path_length = len(url.split('/')[3]) if having_path else 0

    # Having query
    having_query = bool(url.split('?')[1:])  # Check if there is anything after the '?' in the URL

    # Having fragment
    having_fragment = bool(url.split('#')[1:])  # Check if there is anything after the '#' in the URL

    # Having anchor
    having_anchor = bool(url.split('!')[1:])  # Check if there is anything after the '!' in the URL

    # Entropy of URL
    entropy_of_url = 0  # Calculate entropy of the URL (e.g., Shannon entropy)

    # Entropy of domain
    entropy_of_domain = 0  # Calculate entropy of the domain (e.g., Shannon entropy)

    return np.array([url_length, number_of_dots_in_url, having_repeated_digits_in_url, number_of_digits_in_url,
                     number_of_special_char_in_url, number_of_hyphens_in_url, number_of_underline_in_url,
                     number_of_slash_in_url, number_of_questionmark_in_url, number_of_equal_in_url,
                     number_of_at_in_url, number_of_dollar_in_url, number_of_exclamation_in_url,
                     number_of_hashtag_in_url, number_of_percent_in_url, domain_length, number_of_dots_in_domain,
                     number_of_hyphens_in_domain, having_special_characters_in_domain,
                     number_of_special_characters_in_domain, having_digits_in_domain, number_of_digits_in_domain,
                     having_repeated_digits_in_domain, number_of_subdomains, having_dot_in_subdomain,
                     having_hyphen_in_subdomain, average_subdomain_length, average_number_of_dots_in_subdomain,
                     average_number_of_hyphens_in_subdomain, having_special_characters_in_subdomain,
                     number_of_special_characters_in_subdomain, having_digits_in_subdomain,
                     number_of_digits_in_subdomain, having_repeated_digits_in_subdomain, having_path,
                     path_length, having_query, having_fragment, having_anchor, entropy_of_url, entropy_of_domain])


# Example URL for prediction
url = "https://www.google.com.np/"

# Extract features for the example URL
url_features = extract_features(url)

# Make prediction
prediction = model.predict([url_features])[0]

if prediction == 0:
    print("The URL is predicted to be genuine.")
else:
    print("The URL is predicted to be malicious.")
