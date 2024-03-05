from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient, WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from openai import BadRequestError
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data, expand_content)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import os

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)
##############
# START CODE #
##############
data_path = './data/impact_theory_data.json'

## RETRIEVER
api_key = os.environ['WEAVIATE_API_KEY']
url = os.environ['WEAVIATE_ENDPOINT']
client = WeaviateClient(
    api_key, 
    url,
    model_name_or_path='./models/finetuned-all-MiniLM-L6-v2-300',
    # openai_api_key=os.environ['OPENAI_API_KEY']
    )
available_classes=sorted(client.show_classes())
logger.info(available_classes)

## RERANKER
reranker = ReRanker(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

## LLM 
model_ids = ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613']
model_name = model_ids[1]
llm = GPT_Turbo(model=model_name, api_key=os.environ['OPENAI_API_KEY'])

## ENCODING
encoding = encoding_for_model(model_name)

## DATA + CACHE
data_path = 'data/impact_theory_data.json'
cache_path = '/Users/luismi/Downloads/impact_theory_expanded.parquet'
data = load_data(data_path)
cache = None  # Initialize cache as None

# Check if the cache file exists before attempting to load it
if os.path.exists(cache_path):
    cache = load_content_cache(cache_path)
else:
    logger.warning(f"Cache file {cache_path} not found. Proceeding without cache.")

#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main(client: WeaviateClient):
        
    with st.sidebar:
        guest_input = st.selectbox(
            label='Select Guest', 
            options=guest_list, 
            index=None, 
            placeholder='Select Guest'
            )
        alpha_input = st.slider(
            label='Alpha for Hybrid', 
            min_value=0.00, 
            max_value=1.00, 
            value=0.40, 
            step=0.05)
        retrieval_limit = st.slider(
            label='Hybrid Search Retrieval Results', 
            min_value=10, 
            max_value=300, 
            value=10, 
            step=10
            )
        reranker_topk = st.slider(
            label='Reranker Top K',
            min_value=1, 
            max_value=5, 
            value=3, 
            step=1
            )
        temperature_input = st.slider(
            label='Temperature of LLM', 
            min_value=0.0, 
            max_value=2.0, 
            value=0.10, 
            step=0.10
            )
        class_name = st.selectbox(
            label='Class Name:', 
            options=available_classes, 
            index=None,
            placeholder='Select Class Name'
            )

    if class_name == 'Ada_data_256':
        client = WeaviateClient(api_key, url, model_name_or_path='text-embedding-ada-002', openai_api_key=os.environ['OPENAI_API_KEY'])

    client.display_properties.append('summary')
    logger.info(client.display_properties)
    ########################
    ## SETUP MAIN DISPLAY ##
    ########################
    st.image('./assets/impact-theory-logo.png', width=400)
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    col1, _ = st.columns([7,3])
    with col1:
        query = st.text_input('Enter your question: ')
        st.write('\n\n\n\n\n')

        ############
        ## SEARCH ##
        ############
        if query:
            # make hybrid call to weaviate
            guest_filter = WhereFilter(
                path=['guest'],
                operator='Equal',
                valueText=guest_input).todict() if guest_input else None
    
            hybrid_response = client.hybrid_search(
                request=query,
                class_name=class_name,
                alpha=alpha_input,
                display_properties=client.display_properties,
                where_filter=guest_filter,
                limit=retrieval_limit
            )
            # rerank results
            ranked_response = reranker.rerank(
                results=hybrid_response,
                query=query,
                apply_sigmoid=True,
                top_k=reranker_topk
            )
            logger.info(ranked_response)
            expanded_response = expand_content(ranked_response, cache, content_key='doc_id', create_new_list=True)

            # validate token count is below threshold
            token_threshold = 8000 if model_name == model_ids[0] else 3500
            valid_response = validate_token_threshold(
                ranked_results=expanded_response,
                base_prompt=question_answering_prompt_series,
                query=query,
                tokenizer=encoding,
                token_threshold=token_threshold,
                verbose=True
            )

            #########
            ## LLM ##
            #########
            make_llm_call = False
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                # Creates container for LLM response
                chat_container, response_box = [], st.empty()

                # generate LLM prompt
                prompt = generate_prompt_series(query=query, results=valid_response)
                # logger.info(prompt)
                if make_llm_call:

                    try:
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            max_tokens=350, # expand for more verbose answers
                            show_response=True,
                            stream=True):

                            # inserts chat stream from LLM
                            with response_box:
                                content = resp.choices[0].delta.content
                                if content:
                                    chat_container.append(content)
                                    result = "".join(chat_container).strip()
                                    st.write(f'{result}')
                    except BadRequestError:
                        logger.info('Making request with smaller context...')
                        valid_response = validate_token_threshold(
                            ranked_results=ranked_response,
                            base_prompt=question_answering_prompt_series,
                            query=query,
                            tokenizer=encoding,
                            token_threshold=token_threshold,
                            verbose=True
                        )

                        # generate LLM prompt
                        prompt = generate_prompt_series(query=query, results=valid_response)
                        for resp in llm.get_chat_completion(
                            prompt=prompt,
                            temperature=temperature_input,
                            max_tokens=350, # expand for more verbose answers
                            show_response=True,
                            stream=True):

                            try:
                                # inserts chat stream from LLM
                                with response_box:
                                    content = resp.choices[0].delta.content
                                    if content:
                                        chat_container.append(content)
                                        result = "".join(chat_container).strip()
                                        st.write(f'{result}')
                            except Exception as e:
                                print(e)

            ####################
            ## Search Results ##
            ####################
            st.subheader("Search Results")
            for i, hit in enumerate(ranked_response):
                col1, col2 = st.columns([7, 3], gap='large')
                episode_url = hit['episode_url']
                title = hit['title']
                guest=hit['guest']
                show_length = hit['length']
                time_string = convert_seconds(show_length)
                # content = ranked_response[i]['content'] # Get 'content' from the same index in ranked_response
                content = hit['content']
            
                with col1:
                    st.write( search_result(i=i, 
                                            url=episode_url,
                                            guest=guest,
                                            title=title,
                                            content=content,
                                            length=time_string),
                                            unsafe_allow_html=True)
                    st.write('\n\n')

                    with st.expander("Click Here for Episode Summary:"):
                        try:
                            ep_summary = hit['summary']
                            st.write(ep_summary)
                        except Exception as e:
                            print(e)

                with col2:
                    image = hit['thumbnail_url']
                    st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)
                    st.markdown(f'''
                                <p style="text-align: right;">
                                    <b>Episode:</b> {title.split('|')[0]}<br>
                                    <b>Guest:</b> {hit['guest']}<br>
                                    <b>Length:</b> {time_string}
                                </p>''', unsafe_allow_html=True)


if __name__ == '__main__':
    main(client)
