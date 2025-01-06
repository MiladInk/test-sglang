import time
import sglang as sgl
from datasets import load_dataset
from math_grader_minerva import eval_math
from math_answer_extraction import extract_math_answer
from math_extract_steps_inplace import split_solution_inplace


def llm_chat_template(tokenizer, query, assistant_partial_respone=''):
    # you may wonder: why didn't we just use continue_final_message=True? Because when assistant_partial_response is '', the tokenizer adds the <im_end> token :/ idk why.
    instruct_math_eval_prompt = "{query}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruct_math_eval_prompt.format(query=query)},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    chat_query = text + assistant_partial_respone
    
    return chat_query


def is_correct(query, ground_truth_answer, response):
    extracted_answer = extract_math_answer(query, response)
    if extracted_answer is None or len(extracted_answer) == 0:
        return False
    return eval_math(item={'prediction': extracted_answer, 'answer': [ground_truth_answer]}, pred_key='prediction')

def generate_responses(tokenizer, ds):
    sampling_params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 512}
    ds = ds.map(lambda x: {'query': llm_chat_template(tokenizer, query=x['problem'], assistant_partial_respone='')})
    queries_input_ids= ds.map(lambda x: {'query_input_ids': tokenizer(x['query'], add_special_tokens=False)['input_ids']})['query_input_ids']
    responses = llm.generate(input_ids=queries_input_ids, sampling_params=sampling_params) # can't use map here, it should be in one call and sglang handles the batching
    ds = ds.add_column('response', responses)
    return ds

def generated_response_stats(ds):    
    response_tokens_count = ds.map(lambda x: {'y': x['response']['meta_info']['completion_tokens']})['y']
    avg_response_token = sum(response_tokens_count) / len(response_tokens_count)
    all_response_tokens = sum(response_tokens_count)
    
    is_response_correct = ds.map(lambda x: {'y': is_correct(query=x['query'], ground_truth_answer=x['answer'], response=x['response']['text'])})['y']
    avg_correct = sum(is_response_correct) / len(is_response_correct)
    
    return {
        'avg_response_token': avg_response_token,
        'avg_correct': avg_correct,
        'all_response_tokens': all_response_tokens
    }
    

def get_prefixes(response, step_boundaries):
        prefixes = ['',]
        for i in range(len(step_boundaries) - 1):
            prefixes.append(response[step_boundaries[i]:step_boundaries[i+1]])
        return {'response_prefixes': prefixes}
    
def get_prefix_queries(query, prefixes):
    prefix_queries = []
    for prefix in prefixes:
        prefix_queries.append(f"{query}{prefix}")
    return {'prefix_queries': prefix_queries}
    
def get_step_values(query, ground_truth_answer, response_prefixes, prefixes_mcs_outputs):
    step_values = []
    for prefix, outputs in zip(response_prefixes, prefixes_mcs_outputs):
        values = []
        for output in outputs:
            completion = output['text']
            score = is_correct(query=query, ground_truth_answer=ground_truth_answer,response=f"{prefix}{completion}") # TODO: handle no <eos>
            values.append(score) 
        step_values.append(sum(values) / len(values))
    return {'step_values': step_values}
    
def estimate_step_values_by_MC(tokenizer, ds, K: int):
    """
    input: dataset(query, step_boundaries)
    output: dataset(query, step_boundaries, step_values)
    """
    
    ds = ds.map(lambda x: get_prefixes(x['response']['text'], x['step_boundaries']))
    ds = ds.map(lambda x: get_prefix_queries(x['query'], x['response_prefixes']))
    
    # for efficiency, we don't submit thousands of individual requests.
    # we will batch all requests and let sglang handle the scheduling for us.
    # this makes the code a bit dirty to disenangle them afterwards though.
    # we keep track of the counts of prefixes per item. and then split the outputs accordingly.
    all_mc_queries = ds['prefix_queries']
    flatten_mc_queries = []
    count_queries = []
    for mc_queries in all_mc_queries:
        flatten_mc_queries.extend(mc_queries)
        count_queries.append(len(mc_queries))
    
    sampling_params = {"temperature": 0.6, "top_p": 0.9, "max_new_tokens": 512, "n": K}
    flatten_mc_queries_input_ids = tokenizer(flatten_mc_queries, add_special_tokens=False)['input_ids']
    mc_outputs = llm.generate(input_ids=flatten_mc_queries_input_ids, sampling_params=sampling_params)
    
    # first, reshape from nxk to n list of k outputs (sglang just returns a flat list)
    assert len(mc_outputs) == len(flatten_mc_queries) * K
    outputs = []
    for i in range(0, len(flatten_mc_queries)):
        outputs.append(mc_outputs[i*K:i*K+K])
        
    # second, split the outputs per item
    outputs_per_prefix = []
    start = 0
    for count in count_queries:
        outputs_per_prefix.append(outputs[start:start+count])
        start += count
        
    ds = ds.add_column('prefixes_mcs_output', outputs_per_prefix)
    return ds 

def get_step_advantages(values):
    advantages = [values[i] - values[i-1] for i in range(1, len(values))]
    return {'step_advantages': advantages}

def map_step_advantages_to_token_advantages(tokenizer, query, response, step_boundaries, step_advantages):
    # test case for advantage computation
    text = f"{query}{response}"
    
     # building the mask is a bit tricky, where does query tokens end and response tokens start? we assume they are tokenized separately.
    assert tokenizer.encode(query, add_special_tokens=False) + tokenizer.encode(response, add_special_tokens=False) == tokenizer.encode(text, add_special_tokens=False), "one token is both at the end of the query and the beginning of the response"
    # we will build the mask from the position of the last char of the query returned by offset_mapping, as we don't know if the tokenizer adds special tokens.
    
    assert step_boundaries[0] == 0
    step_boundaries = [0] + [x + len(query) for x in step_boundaries[1:]] # step_boundries are relative to the response, we need to adjust them to the text, make query of step 0
    # for i in range(len(step_boundaries)-1):
    #     step_text = text[step_boundaries[i]:step_boundaries[i+1]]
    #     print(f"step {i}th:{step_text}")

    encoded = tokenizer(text,
                        return_offsets_mapping=True,
                        return_tensors="pt")

    offset_mapping = encoded['offset_mapping'][0]
    input_ids = encoded['input_ids'][0]

    # determine which step each token belongs to
    # we know the [start_char_idx, end_char_idx) of each step. Using the offset_mapping, we can determine which step each token belongs to
    # Although close to impossible a token can span two steps, we will assign the advantage of the first step it belongs to.
    token_to_step = []
    total_steps = len(step_boundaries) - 1
    assert total_steps > 0
    step_start_char_idx, step_end_char_idx = step_boundaries[0], step_boundaries[1] # [start_char_idx, end_char_idx)
    step = 0
    for token_start_char_idx, token_end_char_idx in offset_mapping.tolist(): # [start_idx of the char of this token, end_idx of the char of this token)
        if token_start_char_idx < step_end_char_idx:
            assert token_start_char_idx >= step_start_char_idx
        else:
            step += 1
            assert step < total_steps
            step_start_char_idx, step_end_char_idx = step_boundaries[step], step_boundaries[step+1]
            assert token_start_char_idx < step_end_char_idx
            assert token_start_char_idx >= step_start_char_idx, "maybe a token is split across more than two steps, this is terrible"
        
        token_to_step.append(step)
        
    # now, will find the position of the last token of the query
    query_end_char_idx = len(query)-1
    for i, (token_start_char_idx, token_end_char_idx) in enumerate(offset_mapping.tolist()):
        if token_start_char_idx > query_end_char_idx:
            query_end_token_idx = i-1
            break
    else:
        query_end_token_idx = i
    

    assert step == total_steps - 1
    assert len(token_to_step) == len(input_ids)

    # print(f"token_which_step: {token_to_step}")

    # assign the advantages to the tokens
    token_advantages = [0] * len(input_ids) 
    for i in range(len(input_ids)):
        token_advantages[i] = step_advantages[token_to_step[i]]
    
    # print alongside the tokens
    # for (step, token_id, adv) in zip(token_to_step, input_ids, token_advantages):
    #     print(f" token: {tokenizer.decode(token_id)}, step: {step}, advantage: {adv}")  
    
    query_tokens = query_end_token_idx + 1
    response_tokens = len(input_ids) - query_tokens
    mask = [0] * query_tokens + [1] * response_tokens
    
   
    
    return {'token_advantages': token_advantages, 'full_text': text, 'mask': mask, 'full_text_input_ids': input_ids}

def generate_vine_episodes(tokenizer, ds):
    """
    input: dataset['query', 'answer', 'response']
    output: dataset['query', 'answer', 'response', 'full_text_input_ids', 'token_advantages', 'mask'], time taken
    """
    start_time = time.time()
    ds = ds.map(lambda x: {'step_boundaries': split_solution_inplace(x['response']['text'])})
    mc_ds = estimate_step_values_by_MC(tokenizer, ds, K=9)  
    mc_ds = mc_ds.map(lambda x: get_step_values(x['query'], x['answer'], x['response_prefixes'], x['prefixes_mcs_output']))  
    mc_ds = mc_ds.map(lambda x: get_step_advantages(x['step_values']))
    mc_ds = mc_ds.map(lambda x: map_step_advantages_to_token_advantages(llm.get_tokenizer(), x['query'], x['response']['text'], x['step_boundaries'], x['step_advantages']))
    mc_time = time.time() - start_time
    
    return mc_ds, mc_time



if __name__ == '__main__':
    dataset = load_dataset("MathMindsAGI/MATH-openai-split")
    train_ds = dataset["train"]
    llm = sgl.Engine(model_path="Qwen/Qwen2-Math-1.5B-Instruct", dp_size=1)
    start_time = time.time()
    tokenizer = llm.get_tokenizer()
    
    ds = generate_responses(tokenizer, train_ds.select(range(10)))
    
    # stats
    end_time = time.time()
    stats = generated_response_stats(ds)
    token_throughput = stats['all_response_tokens'] / (end_time - start_time)
    print(f"Token throughput: {token_throughput:.2f} tokens per second")
    print(f"Average response token: {stats['avg_response_token']:.2f}")
    print(f"Average correct: {stats['avg_correct']:.2f}")
    
    mc_ds, mc_time = generate_vine_episodes(tokenizer, ds)
    
    # stats
    token_count = 0
    cached_token_count = 0
    for prefixes_mcs_output in mc_ds['prefixes_mcs_output']:
        for prefix_mcs_output in prefixes_mcs_output:
            for mcs_output in prefix_mcs_output:
                token_count += mcs_output['meta_info']['completion_tokens']
                cached_token_count += mcs_output['meta_info']['cached_tokens']
            
    print(f"Token count: {token_count}")
    print(f"Cached token count: {cached_token_count}")
    print(f"MC time: {mc_time:.2f} seconds")
    print(f"Token throughput: {token_count / mc_time:.2f} tokens per second")
    print(f"Cached token throughput: {cached_token_count / mc_time:.2f} tokens per second")
    
    llm.shutdown()






