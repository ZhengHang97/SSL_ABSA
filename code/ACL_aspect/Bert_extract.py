from transformers import *
import numpy as np


# bert
def vectorize_sequences_bert(sequences, model, tokenizer, token_num, dimension=768):
    results = np.zeros((len(sequences), dimension))
    extractBertFeature = pipeline('feature-extraction', model=model, tokenizer=tokenizer, device=0)
    # albert-base-v2  distilbert-base-cased bert-large-cased bert-base-cased xlnet-base-cased albert-xxlarge-v2 longformer-base-4096
    # albert-xlarge-v2 albert-large-v1 albert-large-v2 albert-base-v2  albert-base-v1 bert-large-cased bert-base-cased
    for i, sequence in enumerate(sequences):
        # review_vector = np.zeros(dimension)
        review_vector_tmp = np.zeros(dimension)
        # len_review_words = 0
        sequence = sequence.replace("\xa0", "").replace(".", "").replace("!", "").replace("*", "").replace(",",
                                                                                                           "").replace(
            "(", "").replace(")", "").replace(":", "").replace("-", "").replace(">", "").replace("<", "").replace("=",
                                                                                                                  "").replace(
            "\n", " ").replace("/", "").replace("'", "").replace("?", "").lower()
        # (['!','@','#','$','%','^','&','*','',')','_','-','=','+','\\','|','`','~','[',']','{','}',';',':','\'',
        # '\"',',','.','<','>','/','?'], "")

        if token_num == 300:
            sentence = ' '.join(sequence.split(' '))[:300]  # [:300]
        else:
            sentence = ' '.join(sequence.split(' '))
        print(i)
        review_vector_tmp = np.array(extractBertFeature(sentence))[0][0]
        # try:
        #     review_vector_tmp = np.array(extractBertFeature(sentence))[0][0]
        # except RuntimeError:
        #     print(i)
        #     print(sentence)
        results[i] = review_vector_tmp[0:dimension] * 10

    return results

def encode(text, tokenizer):
    encoded_input = tokenizer.batch_encode_plus(
        text,
        padding=True,
        add_special_tokens=True,
        return_tensors=True,
    )
    return encoded_input