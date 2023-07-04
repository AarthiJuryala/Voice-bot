import os
import streamlit as st
import openai
import pandas as pd
from langchain.llms import OpenAI
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from tempfile import NamedTemporaryFile
import soundfile as sf
import sounddevice as sd
import numpy as np
from streamlit_player import st_player
import base64

from elevenlabs import generate, play, voices
from elevenlabs.api.error import UnauthenticatedRateLimitError, RateLimitError

# Set OpenAI API key


def app():
    st.sidebar.write("Enter the required keys to get an answer.")
    api_key = st.sidebar.text_input("Enter OpenAI API key:")
    elevenlabs_key = st.sidebar.text_input("Enter ElevenLabs key:")
    # Title and description
    st.title("BuddyGPT")

    file_ = open("Images/smallbunny.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="bunny gif">',
        unsafe_allow_html=True,
    )

    audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")


        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file.close()

            # Convert audio to text using speech recognition
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio_file.name) as source:
                audio = r.record(source)
            text = r.recognize_google(audio)

            st.write("Converted Text:")
            st.write(text)

        
        openai.api_key = api_key
        
        
        prompt = f"Question: {text}\nAnswer:"
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None,
            timeout=15,
        )
        answer = response.choices[0].text.strip().split('\n')[0]
        st.write("Generated Answer:")
        st.write(answer)




        # Delete the temporary audio file
        os.remove(temp_audio_file.name)     
        


        try:
            audio = generate(text=answer, voice="Rachel", model='eleven_multilingual_v1',
                            api_key=elevenlabs_key if elevenlabs_key else st.secrets['elevenlabs_key'])

            # Play the audio
            st.audio(audio, format='audio/wav')


        except UnauthenticatedRateLimitError:
            e = UnauthenticatedRateLimitError("Unauthenticated Rate Limit Error")
            st.exception(e)

        except RateLimitError:
            e = RateLimitError('Rate Limit')
            st.exception(e)

        

        
    
if __name__ == "__main__":
    app()
