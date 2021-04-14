"""Generate responses from a dialogue system."""

import argparse
import csv

import constrained_decoding
import util

from collections import defaultdict
from transformers import AutoModelWithLMHead, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file',
                        help='CSV file of prompts to generate from.')
    parser.add_argument('--annotated_file',
                        help='CSV file with annotated samples to use for calculating salient ngrams.')
    parser.add_argument('--output_file',
                        help='File to output generated text.')
    parser.add_argument('--device',
                        default='cpu',
                        help='`cpu` or `cuda`')
    parser.add_argument('--model',
                        default='microsoft/DialoGPT-medium',
                        help='Dialogue model to generate from.')
    parser.add_argument('--tokenizer',
                        default='microsoft/DialoGPT-medium',
                        help='Tokenizer used with dialogue model.')
    parser.add_argument('--seq_len',
                        default=40,
                        type=int,
                        help='How long of a sequence to generate.')
    parser.add_argument('--use_constrained_decoding',
                        action='store_true',
                        help='Whether to use constrained decoding.')
    parser.add_argument('--constrained_decoding_threshold',
                        default=0.0,
                        type=float,
                        help='Threshold for constrained decoding.')
    parser.add_argument('--calc_salience',
                        action='store_true',
                        help='File of salient phrases for ad hom and non-ad homs.')
    parser.add_argument('--salience_sim_threshold',
                        default=5.5,
                        type=float,
                        help='Threshold for salient n-gram similarity.')
    parser.add_argument('--salience_lambda',
                        default=0.5,
                        type=float,
                        help='Lambda to smooth calculation of salient phrases.')
    parser.add_argument('--use_salient_ngrams',
                        action='store_true',
                        help='Use salient ngrams instead of words.')
    parser.add_argument('--use_finetuned',
                        action='store_true',
                        help='Using finetuned model requires extra postprocessing.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Show salient ngrams and decoding operation statistics for constrained decoding.')

    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelWithLMHead.from_pretrained(args.model)
    model = model.to(args.device)

    person_to_category_to_salient_ngram_embed = ()
    word_embeds = ()
    # Use constrained decoding.
    if args.use_constrained_decoding:
        salience_file = '_'.join([args.annotated_file, 'lmbda' + str(args.salience_lambda),
                      'threshold' + str(args.salience_sim_threshold),
                      'ngrams' + str(args.use_salient_ngrams) + '.csv'])

        # Get embeddings of salient ngrams for constrained decoding.
        word_embeds = model.transformer.wte.weight
        person_to_category_to_salient_attrib = constrained_decoding.calc_salient_phrases(
            args.annotated_file,
            salience_file,
            salience_sim_threshold=args.salience_sim_threshold,
            use_ngrams=args.use_salient_ngrams,
            lmbda=args.salience_lambda)
        person_to_category_to_salient_ngram_embed = {
            'persona': defaultdict(list),
            'personb': defaultdict(list)
        }
        for person in person_to_category_to_salient_attrib:
            for category in person_to_category_to_salient_attrib[person]:
                for ngram, salience_val in person_to_category_to_salient_attrib[person][category]:
                    token_idxes = tokenizer.encode(ngram, add_prefix_space=True)
                    ngram_embed = word_embeds[token_idxes, :]
                    person_to_category_to_salient_ngram_embed[person][category].append((ngram, ngram_embed, salience_val))

    # Gather prompts.
    prompts = []
    with open(args.prompt_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        # Format: prompt is the last column of the CSV file.
        for row_idx, row in enumerate(csv_reader):
            prompt = row[-1].strip()
            prompts.append(prompt)
    print('Prompt examples:', prompts[:5])

    # Sample.
    sampling_paths = []
    with open(args.output_file, 'w') as o:
        writer = csv.writer(o, delimiter=',')
        for p in prompts:
            print('=' * 80)

            input_ids = tokenizer.encode(p + tokenizer.eos_token)
            print(tokenizer.decode(input_ids))

            out, sampling_path = constrained_decoding.sample_sequence(
                model=model, length=args.seq_len, context=input_ids, temperature=0.7, top_k=40, device=args.device,
                use_constrained_decoding=args.use_constrained_decoding,
                constrained_decoding_threshold=args.constrained_decoding_threshold,
                person_to_category_to_salient_ngram_embed=person_to_category_to_salient_ngram_embed,
                word_embeds=word_embeds,
                tokenizer=tokenizer)
            sampling_paths.append(sampling_path)
            text = tokenizer.decode(out[:, len(input_ids):][0], skip_special_tokens=False)
            if text.find(tokenizer.eos_token) > 0:
                text = text[:text.find(tokenizer.eos_token)]
            if args.use_finetuned:
                # Cut off generated output at the last ./?/! if there is one.
                text = util.trim_text(text)
            text = text.strip()
            print("DialoGPT: {}".format(text))
            writer.writerow([p, text])

    if args.use_constrained_decoding and args.verbose:
        constrained_decoding.calc_sampling_path_stats(sampling_paths)


if __name__ == '__main__':
    main()
