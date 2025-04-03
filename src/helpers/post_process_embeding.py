import re
import spacy
import difflib
import torch
nlp = spacy.load("en_core_web_sm")

#implemented for batch size 1
def extract_phrase_embeddings(item, model_class):
    """
    Extracts phrase embeddings from the hook data and adds them to the hook_data dictionary.

    Args:
        item (dict): A dictionary containing the phrase and the corresponding embeddings.
        hook_data (dict): A dictionary containing the hook data.

    Returns:
        dict: A dictionary containing the hook data with the phrase embeddings added.
    """
    hook_data = item["hidden_states"]
    tokenizer = model_class.get_tokenizer()
    predicted_text = item["model_predictions"][0]
    doc = nlp(predicted_text)
    phrases = []
    for np in doc.noun_chunks:
       clean_text = clean_string(np.text)
       clean_text = modify_string(clean_text)
       if len(clean_text) > 0:
            phrases.append(clean_text)
    
    
    
    filtered_ebedding_of_hidden_states = {}
    for hidden_key, hidden_value in hook_data.items():
       all_text_tokens_embedding = {}
       for index, num_token in enumerate(item["model_generated_output"][0]):
           token = tokenizer.decode(
               [num_token]
           )
           clean_token = clean_string(token)
           if len(clean_token) > 0:
               all_text_tokens_embedding[clean_token] = hidden_value[:, index: index+1, :]
       filtered_ebedding_of_hidden_states[hidden_key] = all_text_tokens_embedding
            
    new_item = {}
    modified_img_id = []
    modified_instruction = []
    modified_response = []
    modified_image = []
    modified_targets = []
    modified_text = []
    modified_model_output = []
    modified_model_generated_output = []
    modified_predictions = []
    modified_scores = []
    modified_hidden_states = []
    for index, phrase in  enumerate(phrases):
        new_hidden= {}
        for hidden_key, hidden_token_value in filtered_ebedding_of_hidden_states.items():
            phrase_embeddig = []
            #print(tok_ebedding.keys())
 
            tokens_phrase = tokenizer.encode(
            phrase 
                )
            for tok in tokens_phrase:
                t_tok =  tokenizer.decode(tok)
                matches = difflib.get_close_matches(t_tok, hidden_token_value.keys(), n=1, cutoff=0.6)
             
                if len(matches)>0:
                    phrase_embeddig.append(hidden_token_value[matches[0]])
            if len(phrase_embeddig) < 1:
                continue
            stacked = torch.stack(phrase_embeddig, dim=0)  # Shape: [5, 1, 380]
            phrase_embeddig = torch.sum(stacked, dim=0)
            new_hidden[hidden_key] = phrase_embeddig
        
        #modified_img_id.append([f"{phrase}@{item['img_id'][0]}"])
        
        #modified_instruction.append([f"{phrase}@{item['instruction'][0]}"])
        
        
        #modified_response.append([f"{phrase}@{item['response'][0]}"])
        if len(new_hidden) < 1:
            continue
        if "image" not in item: 
            modified_image.append(item['text'])
            modified_predictions.append(item['model_predictions'])
        else:
            modified_image.append([f"{phrase}@{item['image'][0]}"])
            modified_predictions.append([f"{phrase}@{item['model_predictions'][0]}"])
        #modified_targets.append([f"{phrase}@{item['targets'][0]}"])
        #modified_text.append([f"{phrase}@{item['text'][0]}"])
        #modified_model_output.append(item['model_output'])
        #modified_model_generated_output.append(item['model_generated_output'])
        
        #modified_scores.append([item['scores']])
        
        modified_hidden_states.append(new_hidden)
        
 

    #new_item["img_id"] = modified_img_id
    #new_item["instruction"] = modified_instruction
    #new_item["response"] = modified_response
    if "image" not in item: 
        new_item["text"] = modified_image
    else:
        new_item["image"] = modified_image
    #new_item["targets"] = modified_targets
    #new_item["text"] = modified_text
    #new_item["model_output"] = modified_model_output
    #new_item["model_generated_output"] = modified_model_generated_output
    new_item["model_predictions"] = modified_predictions
    #new_item["scores"] = modified_scores
    new_item["hidden_states"] = modified_hidden_states
        

    return new_item


# Clean string and remove noun pronouns articles, prepositions, and conjunctions
def clean_string(input_string, remove_words=None):
    # Define a default list of words to remove if none are provided
    if remove_words is None:
        remove_words = []
        
    # List of English articles
    articles = r'\b(?:a|an|the)\b'
    
    # List of English pronouns
    pronouns = r'\b(?:i|you|he|she|it|we|they|me|him|her|us|them|my|mine|your|yours|his|hers|its|our|ours|their|theirs|myself|yourself|himself|herself|itself|ourselves|yourselves|themselves|photo|image|question)\b'

    # Combine articles and pronouns into one regex pattern
    combined_pattern = f"{articles}|{pronouns}"
    
    # Remove articles and pronouns
    input_string = re.sub(combined_pattern, '', input_string, flags=re.IGNORECASE)
    
    # Remove specific words if provided
    for word in remove_words:
        input_string = re.sub(rf'\b{re.escape(word)}\b', '', input_string, flags=re.IGNORECASE)
    
    # Remove special characters
    input_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    
    # Remove extra spaces
    return re.sub(r'\s+', ' ', input_string).strip()

def modify_string(input_string):
    if input_string and input_string[0].islower():
        return ' ' + input_string
    return input_string