# ad-hom-in-dialogue

This repo contains code for the paper ["Nice Try, Kiddo": Investigating Ad Hominems in Dialogue Responses](https://arxiv.org/abs/2010.12820) (NAACL 2021).


## Data
The full AdHomInTweets dataset consists 14.5K [post, response] English Tweet pairs. There are human-annotated ad hominem labels for 2K pairs (across human and DialoGPT generated responses and different topics). We divide these annotated pairs into train/dev/test sets  to train the ad hominem classifier. Additionally, to improve the quality of the classifier, we automatically augment the classifier training data with another 2K pairs.

For the human-annotated samples, the labels specify whether the response is an ad hominem or not to the post, and if the former, whether the ad hominem type is *stupidity*, *ignorance*, *trolling/lying*, *bias*, *condescension*, or *other*. For the classifier-labeled samples, the labels are a binary categorization of whether the response has an ad hominem(s) to the post.
See the original paper for more category details.

To comply with Twitter's [Terms of Service](https://developer.twitter.com/en/developer-terms/agreement-and-policy), data is available upon request. Please contact Emily at ewsheng at isi dot edu.

## Setup
```
conda create --name ad-hominems python==3.7
conda activate ad-hominems
pip install -r requirements.txt
```

## Ad Hominem Classifier
To run the trained ad hominem classifier, first download the model here: [coming soon].

Then use the following command:
```
python run_adhom_classifier.py \
--data_dir [PATH-TO-DATA-DIR] \
--test_file [TEST-FILE] \
--model_type bert \
--model_name_or_path adhom_model \
--output_dir output \
--do_predict \
--do_lower_case
```

## Constrained Decoding: SalienSimTop-k
Coming soon
