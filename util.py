"""pre/post processing functions."""

import csv
import logging
import numpy as np
import os
import re
import string

logger = logging.getLogger(__name__)


class InputExample(object):
	"""A single training/test example for ad hominem classification."""

	def __init__(self, guid, persona, personb, src, hashtag, label=None):
		"""Constructs a InputExample.

		Args:
		  guid: Unique id for the example.
		  persona: string. The untokenized text of the first sequence.
		  personb: string. The untokenized text of the seconed sequence.
		  src: string.
		  label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
		self.guid = guid
		self.persona = persona
		self.personb = personb
		self.src = src
		self.hashtag = hashtag
		self.label = label


class InputFeatures(object):
	"""A single set of features of data."""
	def __init__(self,
				   input_ids,
				   input_mask,
				   segment_ids,
	               src,
	               hashtag,
				   label_id):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.src = src
		self.hashtag = hashtag
		self.label_id = label_id


def read_examples_from_csv_file(data_dir, data_file, is_test=False):
	file_path = os.path.join(data_dir, data_file)
	guid_index = 1
	examples = []
	with open(file_path, encoding="utf-8") as f:
		reader = csv.reader(f, delimiter=',')
		# Train/dev format: label,response_source,hashtag,persona,personb.
		# Test format: persona,personb.
		for row in reader:
			persona = row[-2].strip().split()
			personb = row[-1].strip().split()
			if not is_test:
				label = row[0]
				src = row[1]
				hashtag = row[2]
			else:
				label = '1'
				src = 'dialogpt'
				hashtag = '#placeholder'
			examples.append(InputExample(guid="%s-%d".format(data_file, guid_index),
										 persona=persona,
			                             personb=personb,
			                             hashtag=hashtag,
			                             src=src,
										 label=label))
	if not is_test:
		np.random.shuffle(examples)
	return examples


def convert_examples_to_features(examples,
								 label_list,
								 max_seq_length,
								 tokenizer,
								 cls_token_at_end=False,
								 cls_token="[CLS]",
								 cls_token_segment_id=1,
								 sep_token="[SEP]",
								 sep_token_extra=True,
								 pad_on_left=False,
								 pad_token=0,
								 pad_token_segment_id=0,
								 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
								 mask_padding_with_zero=True):
	""" Loads a data file into a list of `InputBatch`s
		`cls_token_at_end` define the location of the CLS token:
			- False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
			- True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
		`cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
	"""

	label_map = {label: i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d", ex_index, len(examples))

		label_id = label_map[example.label]

		tokens_a = []
		for word in example.persona:
			tokens_a.extend(tokenizer.tokenize(word))
		segment_ids_a = [sequence_a_segment_id] * len(tokens_a)

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0   0   0   0  0     0   0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.

		special_tokens_count = 3 if sep_token_extra else 2
		if sep_token_extra:
			tokens_b = []
			segment2 = example.personb
			for word in segment2:
				tokens_b.extend(tokenizer.tokenize(word))
			segment_ids_b = [sequence_b_segment_id] * len(tokens_b)

			# Account for [CLS] and [SEP] with "- 3" for adhom.
			if len(tokens_b) > max_seq_length - special_tokens_count:
				tokens_a = []
				segment_ids_a = []
				tokens_b = tokens_b[:(max_seq_length - special_tokens_count)]
				segment_ids_b = segment_ids_b[:(max_seq_length - special_tokens_count)]
			elif len(tokens_a + tokens_b) > max_seq_length - special_tokens_count:
				# Remove tokens from the end of tokens_a.
				tokens_a = tokens_a[:(max_seq_length - special_tokens_count - len(tokens_b))]
				segment_ids_a = segment_ids_a[:(max_seq_length - special_tokens_count - len(tokens_b))]
		else:
			# Account for [CLS] and [SEP] with "- 3" for adhom.
			if len(tokens_a) > max_seq_length - special_tokens_count:
				# Remove tokens from the end of tokens_a.
				tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
				segment_ids_a = segment_ids_a[:(max_seq_length - special_tokens_count)]

		tokens = tokens_a + [sep_token]
		segment_ids = segment_ids_a + [sequence_a_segment_id]
		if sep_token_extra:
			tokens += tokens_b + [sep_token]
			segment_ids += segment_ids_b + [sequence_b_segment_id]

		if cls_token_at_end:
			tokens += [cls_token]
			segment_ids += [cls_token_segment_id]
		else:
			tokens = [cls_token] + tokens
			segment_ids = [cls_token_segment_id] + segment_ids

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding_length = max_seq_length - len(input_ids)
		if pad_on_left:
			input_ids = ([pad_token] * padding_length) + input_ids
			input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
			segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
		else:
			input_ids += ([pad_token] * padding_length)
			input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
			segment_ids += ([pad_token_segment_id] * padding_length)

		if len(input_ids) != max_seq_length:
			print(tokens)
			print(len(input_ids), max_seq_length)
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("guid: %s", example.guid)
			logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
			logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
			logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
			logger.info("hashtag: %s", example.hashtag)
			logger.info("src: %s", example.src)
			logger.info("label_id: %s", str(label_id))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
				              hashtag=example.hashtag,
				              src=example.src,
							  label_id=label_id))
	return features


def get_labels():
	"""Binary ad hominem labels."""
	return ['0', '1']


def trim_text(text):
	"""Helper method to trim generated text."""
	# Cut off generated output at the last ./?/! if there is one,
	# unless the text ends with hashtags and the last punc is before the hashtags.
	end_punc = '.!?'
	max_end_idx = -1
	for end in end_punc:
		end_idx = text.rfind(end)
		if end_idx > max_end_idx:
			max_end_idx = end_idx
	if max_end_idx == -1:
		return text
	else:
		if max_end_idx + 1 < len(text) and '#' in text[max_end_idx + 1:]:
			return text
		elif max_end_idx + 2 < len(text) and text[max_end_idx + 1] == '"':
			return text[:max_end_idx + 2]
		else:
			return text[:max_end_idx + 1]


def clean_text(text, is_tweet=False):
	"""Clean text to calculate salient ngrams."""
	text = text.lower()
	text = text.encode('ascii', 'ignore').decode()
	if is_tweet:
		text = clean_tweet(text)
	for p in string.punctuation:  # Remove punctuation.
		if p in "'<>|":
			continue
		text = text.replace(p, '')
	text = re.sub('\\s+', ' ', text)
	return text


def clean_tweet(tweet):
	"""Processing if the text is a tweet."""
	# Remove twitter Return handles (RT @xxx:).
	tweet = _filter_retweets(tweet)
	# Remove URL links (httpxxx).
	tweet = _filter_urls(tweet)
	# Remove twitter handles (@xxx).
	tweet = _filter_usernames(tweet)
	# Remove emails.
	tweet = _filter_emails(tweet)
	# Remove hashtags.
	tweet = _filter_hashtags(tweet)
	tweet = _mod_pattern('\n', ' ', tweet)
	tweet = _mod_pattern('  ', ' ', tweet)
	tweet = tweet.strip()
	return tweet


def _mod_pattern(pattern, replace, input_txt):
	"""Find all instances of pattern and possibly replace."""
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(re.escape(i), replace, input_txt)
	return input_txt


def _filter_retweets(text):
	pattern = r"RT @[\w]*:"
	text = _mod_pattern(pattern, '', text)
	text = text.replace('  ', ' ').strip()
	return text


def _filter_urls(text):
	pattern = r"https?://[A-Za-z0-9./]*"
	text = _mod_pattern(pattern, '', text)
	text = text.replace('  ', ' ').strip()
	return text


# https://stackoverflow.com/questions/2304632/regex-for-twitter-username
def _filter_usernames(text):
	pattern = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"
	text = _mod_pattern(pattern, '', text)
	text = text.replace('  ', ' ').strip()
	return text


def _filter_emails(text):
	pattern = r"\S+@\S+"
	text = _mod_pattern(pattern, '', text)
	text = text.replace('  ', ' ').strip()
	return text


def _filter_hashtags(text):
	"""Replace all hashtags with #[hashtag]."""
	pattern = r"#[\w]+"
	text = _mod_pattern(pattern, '', text)
	text = text.replace('  ', ' ').strip()
	return text
