"""Methods for constrained decoding."""

import csv
import ngram
import os
import torch

import util

from collections import defaultdict, Counter
import torch.nn.functional as F


def calc_salient_phrases(data_file, output_file, salience_sim_threshold=5.5, use_ngrams=True, lmbda=0.5):
	"""Given file of different annotated categories, calculate salient phrases for each category.

	:param data_file: CSV file of label,persona,personb.
	:param output_file: file to output salient phrase calculations.
	:param salience_sim_threshold:
	:param use_ngrams: whether to use ngrams or just words for salience calc.
	:param lmbda:
	:return:
		dict of person to category to salient attributes.
	"""
	# Note: this method calculates salient phrases for persona as well as personb.
	# To reduce ad hominem responses, we only use salient phrases from personb.

	if os.path.exists(output_file):
		person_to_category_to_salient_attrib = {
			'persona': defaultdict(list),
			'personb': defaultdict(list)
		}
		with open(output_file, 'r') as f:
			reader = csv.reader(f, delimiter=',')
			for row in reader:
				person = row[0]
				label = row[1]
				ngrm = row[2]
				score = row[3]
				person_to_category_to_salient_attrib[person][label].append((ngrm, score))

		return person_to_category_to_salient_attrib

	# Organize text.
	person_to_category_to_text = {
		'persona': defaultdict(set),
		'personb': defaultdict(set)
	}
	with open(data_file, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for row_idx, row in enumerate(reader):
			labels = row[0]
			persona = util.clean_text(row[-2], is_tweet=True)
			personb = util.clean_text(row[-1], is_tweet=True)
			for label in labels:
				person_to_category_to_text['persona'][label].add(persona)
				person_to_category_to_text['personb'][label].add(personb)

	# Organized salient attributes.
	person_to_category_to_attrib_count = {
		'persona': defaultdict(Counter),
		'personb': defaultdict(Counter)
	}
	for person in person_to_category_to_text:
		category_to_text = person_to_category_to_text[person]
		for category in category_to_text:
			for text in category_to_text[category]:
				words = text.split()
				if use_ngrams:  # [3,4,5]-grams.
					three_gram_index = ngram.NGram(N=3)
					three_grams = list(three_gram_index.ngrams(words))
					four_gram_index = ngram.NGram(N=4)
					four_grams = list(four_gram_index.ngrams(words))
					five_gram_index = ngram.NGram(N=5)
					five_grams = list(five_gram_index.ngrams(words))
					seen = set()
					for gram in three_grams + four_grams + five_grams:
						joined_gram = ' '.join(gram)
						if joined_gram not in seen:
							person_to_category_to_attrib_count[person][category][joined_gram] += 1
							seen.add(joined_gram)
				else:
					seen = set()
					for word in words:
						if word not in seen:
							person_to_category_to_attrib_count[person][category][word] += 1
							seen.add(word)

	# Calculate salience for ad hom vs not (for persona and personb).
	person_to_category_to_salient_attrib = {
		'persona': defaultdict(list),
		'personb': defaultdict(list)
	}
	for person in person_to_category_to_attrib_count:
		category_to_grams = person_to_category_to_attrib_count[person]
		yes_gram_counts = category_to_grams['1']
		total_yes_grams = sum(yes_gram_counts.values())
		no_gram_counts = category_to_grams['0']
		total_no_grams = sum(no_gram_counts.values())
		factor = total_no_grams / total_yes_grams
		for yes_gram in yes_gram_counts:
			if yes_gram in no_gram_counts:
				# Note: we want to normalize ngram counts for '1' and '0', but also be able to operate at an intuitive magnitude.
				# Let's use ngram counts for '1' as the base and scale '0' counts relative to those for '1'.
				salience_val = (yes_gram_counts[yes_gram] + lmbda) / (no_gram_counts[yes_gram] / factor + lmbda)
			else:
				salience_val = (yes_gram_counts[yes_gram] + lmbda) / lmbda
			if salience_val >= salience_sim_threshold:
				person_to_category_to_salient_attrib[person]['1'].append((yes_gram, salience_val))
			elif salience_val <= - salience_sim_threshold:
				person_to_category_to_salient_attrib[person]['0'].append((yes_gram, salience_val))
		for no_gram in no_gram_counts:
			if no_gram not in yes_gram_counts:
				salience_val = (no_gram_counts[no_gram] / factor + lmbda) / lmbda
				if salience_val >= salience_sim_threshold:
					person_to_category_to_salient_attrib[person]['0'].append((no_gram, salience_val))

	# Save salient attributes to file.
	with open(output_file, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		attributes = []
		for person in person_to_category_to_salient_attrib:
			category_to_salient_attrib = person_to_category_to_salient_attrib[person]
			for category in category_to_salient_attrib:
				salience_attrib_list = category_to_salient_attrib[category]
				salience_attrib_list = sorted(salience_attrib_list, key=lambda x: x[1], reverse=True)
				for gram, val in salience_attrib_list:
					attributes.append([person, category, gram, val])
		writer.writerows(attributes)

	return person_to_category_to_salient_attrib


def get_recent_ngrams(curr_output, word_embeds, tokenizer):
	"""Get the most recent ngrams to calculate semantic similarity with salient ngrams."""
	check_eos_token_id_bool = curr_output[0, :] == tokenizer.eos_token_id
	check_eos_token_id = torch.nonzero(check_eos_token_id_bool.int())
	eos_idx = -1
	if len(check_eos_token_id) > 0:
		eos_idx = check_eos_token_id[0][0].item()
	five_start_idx = curr_output.size()[-1] - 5
	gram_start_idx = max(eos_idx, five_start_idx)
	five_gram = curr_output[:, gram_start_idx:]
	five_gram_embeds = torch.mean(word_embeds[five_gram], dim=1, keepdim=True)
	five_gram_embeds = torch.squeeze(five_gram_embeds, dim=0)
	return five_gram_embeds


# https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
def sim_matrix(a, b, eps=1e-8):
	"""Cosine similarity."""
	# Given that cos_sim(u, v) = dot(u, v) / (norm(u) * norm(v))
	#                          = dot(u / norm(u), v / norm(v))
	# We fist normalize the rows, before computing their dot products via transposition:
	a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
	a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
	b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
	sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
	return sim_mt


def semantic_similarity(embeds1, embeds2):
	"""Calculate the semantic similarity between embeds1 and embeds2.

	:param embeds1:
	:param embeds2:
	:return:
	"""
	sim = sim_matrix(embeds1, embeds2)
	return torch.mean(sim)


def constraint(curr_output, log_probs, k_sample_history, threshold, sampling_path, backtrack,
			   person_to_category_to_salient_ngram_embed, word_embeds, past, tokenizer, device):
	"""Determine whether curr_output is similar to salient_attrib_a and dissimilar
		to salient_attrib_b within a threshold.

	:param curr_output: current generated output
	:param log_probs:
	:param k_sample_history:
	:param threshold:
	:param sampling_path:
	:param backtrack:
	:param person_to_category_to_salient_ngram_embed:
	:param word_embeds:
	:param past:
	:param tokenizer:
	:param device:
	:return:
	"""
	sampled_tokens = torch.multinomial(log_probs, num_samples=10, replacement=True)
	all_eos = torch.all(torch.eq(sampled_tokens, tokenizer.eos_token_id))

	# Iterate through sampled token candidates.
	for sampled_token in sampled_tokens[0]:
		sampled_token = torch.reshape(sampled_token, (1, 1))
		new_output = torch.cat((curr_output, sampled_token), dim=1)

		# Special forced choice conditions:
		# If we've reached the backtrack limit
		# or the sampled tokens are all EOS
		# or we've just started generation.
		if backtrack > 5 or all_eos or (k_sample_history.size()[0] < 3 and len(sampling_path) < 3):
			sampling_path.append((k_sample_history.size()[0], sampled_token))
			k_sample_history = torch.cat((k_sample_history, log_probs), dim=0)
			return sampled_token, new_output, k_sample_history, backtrack, past

		# Take the most recent 5-gram and compare sem sim with salient_attrib_a and
		# salient_attrib_b to determine if we should continue generating or replace/choose diff path.
		new_output_tokenized = tokenizer.encode(util.clean_text(tokenizer.decode(new_output[0, :])))
		new_output_tokenized = torch.tensor(new_output_tokenized, device=device, dtype=torch.long).unsqueeze(0)
		curr_gram_embeds = get_recent_ngrams(new_output_tokenized, word_embeds, tokenizer)
		undesired_ngram_embeds = person_to_category_to_salient_ngram_embed['personb']['1']
		desired_ngram_embeds = person_to_category_to_salient_ngram_embed['personb']['0']
		salient_attrib_a = torch.stack([torch.mean(x[1], keepdim=False, dim=0) for x in undesired_ngram_embeds])
		salient_attrib_b = torch.stack([torch.mean(x[1], keepdim=False, dim=0) for x in desired_ngram_embeds])
		sim_a = semantic_similarity(curr_gram_embeds, salient_attrib_a)
		sim_b = semantic_similarity(curr_gram_embeds, salient_attrib_b)

		if sim_a - sim_b <= threshold:  # Found a good candidate.
			sampling_path.append((k_sample_history.size()[0], sampled_token))
			k_sample_history = torch.cat((k_sample_history, log_probs), dim=0)
			return sampled_token, new_output, k_sample_history, backtrack, past

	# Could not find a good candidate at this timestep.
	# Unroll to the previous timestep if possible.
	if k_sample_history.size()[0] >= 1:
		past = [x[:, :, :, :-1, :] for x in past]
		sampled_token, new_output, k_sample_history, backtrack, past = constraint(
			curr_output[:, :-1], k_sample_history[-1:, :], k_sample_history[:-1, :],
			threshold, sampling_path, backtrack, person_to_category_to_salient_ngram_embed,
			word_embeds, past, tokenizer, device)
		backtrack += 1
		return sampled_token, new_output, k_sample_history, backtrack, past
	else:  # No good next candidates and no previous tokens to backtrack to. Randomly choose a top token and go forward.
		sampled_token = sampled_tokens[0][0]
		sampled_token = torch.reshape(sampled_token, (1, 1))
		new_output = torch.cat((curr_output, sampled_token), dim=1)
		sampling_path.append((k_sample_history.size()[0], sampled_token))
		k_sample_history = torch.cat((k_sample_history, log_probs), dim=0)
		return sampled_token, new_output, k_sample_history, backtrack, past


def sampling(output, log_probs, k_sample_history, use_constrained_decoding,
			 constrained_decoding_threshold=0.3, sample=True,
			 sampling_path=(), backtrack=0, person_to_category_to_salient_ngram_embed=(), word_embeds=(), past=None,
			 tokenizer=None, device='cuda'):
	"""Sample using constrained decoding or multinomial or return top-1."""
	if use_constrained_decoding:
		prev, output, k_sample_history, backtrack, past = constraint(
			output, log_probs, k_sample_history, constrained_decoding_threshold, sampling_path, backtrack,
			person_to_category_to_salient_ngram_embed, word_embeds, past, tokenizer, device)
	else:
		if sample:
			prev = torch.multinomial(log_probs, num_samples=1)
		else:
			_, prev = torch.topk(log_probs, k=1, dim=-1)
		output = torch.cat((output, prev), dim=1)
	return prev, output, k_sample_history, backtrack, past


def top_k_logits(logits, k):
	"""Get the score for the top-k logits.

	:param logits:
	:param k:
	:return:
	"""
	if k == 0:
		return logits
	values = torch.topk(logits, k)[0]
	batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
	return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def sample_sequence(model, length, context=None, temperature=1.0, top_k=10, sample=True,
					device='cuda', use_constrained_decoding=False, constrained_decoding_threshold=0.3,
					person_to_category_to_salient_ngram_embed=(), word_embeds=(), tokenizer=None):
	"""

	:param model:
	:param length:
	:param context:
	:param temperature:
	:param top_k:
	:param sample:
	:param device:
	:param use_constrained_decoding:
	:param constrained_decoding_threshold:
	:param person_to_category_to_salient_ngram_embed:
	:param word_embeds:
	:param tokenizer:
	:return:
	"""
	# Assume batch size of 1.
	context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0)
	orig_context_length = context.size()[-1]
	prev = context
	output = context
	past = None
	k_sample_history = torch.tensor([], device=device, dtype=torch.float)
	sampling_path = []  # List of (timestep, token)s tried. Could be moving forward, alternate, or backward in timestep.
	backtrack = 0
	with torch.no_grad():
		while output.size()[-1] < orig_context_length + length:
			# when using `past`, the context for the next call should be only
			# the previous token: https://github.com/huggingface/transformers/issues/1749
			logits, past = model(prev, past=past)
			logits = logits[:, -1, :] / temperature
			logits = top_k_logits(logits, k=top_k)
			log_probs = F.softmax(logits, dim=-1)
			prev, output, k_sample_history, backtrack, past = sampling(
				output, log_probs, k_sample_history, use_constrained_decoding, constrained_decoding_threshold, sample,
				sampling_path, backtrack, person_to_category_to_salient_ngram_embed, word_embeds, past, tokenizer, device)
			if prev == tokenizer.eos_token_id:
				break
	return output, sampling_path


def calc_sampling_path_stats(sampling_paths):
	"""Calculate statistics for decoding operations taken for generating a sample.

	:param sampling_paths:
	:return:
	"""
	stats = Counter()
	for path in sampling_paths:
		curr_step = -1
		for step, token in path:
			if step == curr_step:
				# Picking alternatives at the same timestep.
				stats['alternative'] += 1
			elif step > curr_step:
				# Moving forward in timestep.
				stats['forward'] += 1
			else:
				# Backtracking.
				stats['backward'] += 1
			curr_step = step
	total = sum(stats.values())
	for stat in stats:
		print('Operation:', stat, stats[stat] / total)
