import streamlit as st
import parameters
import tools
from model import *
import torch

def display_predict_page(model):
    st.title('Friends(~like) Script Generator')
    st.info("This model was designed to produce new scripts of the tv show 'Friends'. It was trained on the scripts of seasons 1 to 8 episodes.")


    context = ''
    context += st.text_input('Context for the sequence (may leave blank):')
    context += '\n'

    col1, col2 = st.columns([1,4])
    max_tokens = col1.text_input('Size of generated text', value='400')
    try:
        max_tokens = int(max_tokens)
        assert(max_tokens > 0 and max_tokens < 3001)
    except ValueError:
        st.error('Please enter an integer number.')
    except TypeError:
        st.error('Please enter an integer number.')
    except AssertionError:
        st.error('Please enter an integer between 1 and 3000.')

    button = st.button('Generate Sequence')
    #Creating a Friends Sequence
    if button:
        input = torch.tensor(tools.encode(context), dtype=torch.long).view(1,len(context)).to(parameters.device)
        generated_text = tools.decode(model.generate(idx = input, max_new_tokens=max_tokens)[0].tolist())
        st.text(generated_text)

