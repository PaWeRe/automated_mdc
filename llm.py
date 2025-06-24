import torch
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from tqdm import tqdm
import re

''' 
    ZERO-SHOT TEXT CLASSIFICATION FOR CLINICAL KEYWORD EXTRACTION
    - 2 approaches:
        1. nli_keyword_extractor() - based on natural language inference approach with BART model
        2. es_keyword_extractor() - based on similarity of embeddings created by BERT model
'''

def es_img_preprocessing(
        descriptions: list
    ) -> list:
    ''' 
        desc
            To increase semantic difference between candidates to 
            increase model performance wihtout retraining / finetuning
        args
        returns    
    '''
    full_descriptions = []
    # increase difference between terms by transforming abbreviations into complete descriptions
    abbreviations = {
        'ax': 'axial', 
        'sag': 'sagittal',
        'cor': 'coronal',
        't2': 't two produced by using longer TE and TR times', #https://case.edu/med/neurology/NR/MRI%20Basics.htm#:~:text=T2%20(transverse%20relaxation%20time)%20is,MRI%20IMAGING%20SEQUENCES
        't1': 't one produced by using short TE and TR times',
        'dwi': 'diffusion coefficient weighted imaging',
        'd': 'diffusion coefficient',
        '1400': 'one thousand four hundred',
        '500': 'five hundred',
        'mm2': 'millimeter squared',
        'adc': 'apparent diffusion coefficient',
        'bvalue': 'bvalue factor that reflects the strength and timing of the gradients used to generate diffusion-weighted images' # https://mriquestions.com/what-is-the-b-value.html
        }
    # normalize desriptions
    for description in descriptions:
        description = description.lower() 
        description = re.sub(r'[^\w\s]', '', description) 
        description = re.sub(r'[-_]', ' ', description) 
        description = re.sub(r'\s+', ' ', description).strip() 
        # add more context and specificity by defining static mapping of abbreviations
        words = description.split()
        expanded_words = []
        for word in words:
            if word in abbreviations:
                expanded_words.extend(abbreviations[word].split())
            else:
                expanded_words.append(word)
        full_description = ' '.join(expanded_words)
        full_descriptions.append(full_description)
    return full_descriptions

def es_dsection_preprocessing(
        d_sections: list
    ) -> list:
    ''' 
        desc
            To increase semantic difference between candidates to 
            increase model performance wihtout retraining / finetuning
        args
        returns      
    '''
    d_sections_processed = []
    split_words = {
            'POSTEROMEDIAL': 'POSTERIOR MEDIAL',
            'POSTERIOMEDIAL': 'POSTERIOR MEDIAL',
            'POSTERIORMEDIAL': 'POSTERIOR MEDIAL',
            'POSTEROLATERAL': 'POSTERIOR LATERAL',
            'POSTERIOLATERAL': 'POSTERIOR LATERAL',
            'POSTERIORLATERAL': 'POSTERIOR LATERAL',
            'ANTEROLATERAL': 'ANTERIOR LATERAL',
            'ANTERIOLATERAL': 'ANTERIOR LATERAL',
            'ANTERIORLATERAL': 'ANTERIOR LATERAL'
            }
    for d_section in d_sections:
        # shorten string
        if ':' in d_section:
            first_part = d_section.split(':')[0]
            #8 catches NOTE, Comment, newline, **** edge cases
            if len(first_part) > 8 and first_part.isupper() and '*' not in first_part: 
                d_section = first_part
        else:
            d_section = d_section[:80]
        # standardize anatomical zone description
        for word, split_word in split_words.items():
            if word in d_section:
                d_section = d_section.replace(word, split_word)
        d_sections_processed.append(d_section)
    return d_sections_processed

def get_vector_representation(
        tokenizer,
        model,
        text: str,
        ):
        ''' 
            desc
                Define function to get vector representation of text
            args
            returns
        '''
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            embeddings = outputs[0][0][1:-1]
        return embeddings.mean(axis=0)

def nli_keyword_extractor(
        premises: list,
        hypotheses: list,
        model_id: str,
        threshold: float,
    ) -> dict:
    ''' 
        desc
            Use entailment of given labels / hypotheses and given sentences / premises
            as way to classify radiology impression sections into a specific PI-RADS
            score, Lesion Size and anatomical region. For this task it is important that
            every hypotheses finds a corresponding premise or phrased the traditional way
            every premise should have a hypothesis (and additionally every hypothesis should
            have a premise in our case).
        args
        return
    '''
    best_match = {}
    all_premises = {}
    num_hypotheses = len(hypotheses)
    error_dict = {'no_hypotheses_or_no_premises': ['no_hypotheses_or_no_premises', -1.0]}
    classifier = pipeline("zero-shot-classification",model=model_id)
    if not hypotheses:
        print(f'No hypotheses given!')
        return error_dict
    for premise in tqdm(premises):
        if premise != '':
            result = classifier(premise, hypotheses, multi_label=False)
            best_hypothesis = result['labels'][0]
            best_score = result['scores'][0]
            all_premises[premise] = [best_hypothesis, best_score]
    # if all premises fail
    if not all_premises:
        print(f'All premises empty!')
        return error_dict
    # store only premises with highest entailment score (based on number of lablels)
    sorted_keys = sorted(all_premises, key=lambda k: all_premises[k][1], reverse=True)
    top_m_keys = sorted_keys[:num_hypotheses]
    best_match = {
        hypothesis: [key, all_premises[key][1]] 
        for hypothesis in hypotheses for key in top_m_keys 
        if all_premises[key][0] == hypothesis
    }
    # if the score is < .5, probably no correct diagnosis so discard (contradict hypotheses)
    if best_match[list(best_match.keys())[0]][1] < threshold:
        new_key = 'no_match'
        best_match = {new_key: v for k, v in best_match.items()}
    return best_match
   
def es_keyword_extractor(
        candidates: list,
        targets: list,
        model_id: str,
        threshold: float,
        d_metric: str,
        augment_targets: str
    ) -> dict:
    ''' 
        desc
        args
        return
            - dict with target, best candidate (if at all) for target and confidence score
    '''
    best_sim = -1
    best_match = {}
    targets_processed = []
    candidates_processed = []
    # set distance metric (configurabel in YAML)
    if d_metric == 'cosine':
        d_metric = cosine
    # augmentation technqiues (optional)
    if augment_targets == 'series_descr_augmentation':
        candidates_processed = es_img_preprocessing(candidates)
        targets_processed = es_img_preprocessing(targets)
    elif augment_targets == 'd_section_augmentation':
        candidates_processed = es_dsection_preprocessing(candidates)
        targets_processed = es_dsection_preprocessing(targets)
    else:
        candidates_processed = candidates
        targets_processed = targets
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for i,target in tqdm(enumerate(targets_processed), position=0):
        for j,candidate in tqdm(enumerate(candidates_processed), position=1):
            candidate_vec = get_vector_representation(tokenizer, model, candidate)
            target_vec = get_vector_representation(tokenizer, model, target)
            # TODO: recheck correct implementation of cosine distance / cosine similarity
            sim_score = 1-d_metric(target_vec, candidate_vec)
            if (sim_score > best_sim) and (sim_score > threshold):
                best_sim = sim_score
                # return unaugmented candidates and targets with i,j
                best_match[targets[i]] = [candidates[j], best_sim]
        best_sim = -1 # reset for new target_vec
    return best_match