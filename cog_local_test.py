import requests
import base64
import os

url = "http://localhost:5000/predictions"

reference1 = (
    "https://github.com/sh-lee-prml/HierSpeechpp/raw/main/example/reference_1.wav"
)
reference2 = (
    "https://github.com/sh-lee-prml/HierSpeechpp/raw/main/example/reference_2.wav"
)
reference3 = (
    "https://github.com/sh-lee-prml/HierSpeechpp/raw/main/example/reference_3.wav"
)
reference4 = (
    "https://github.com/sh-lee-prml/HierSpeechpp/raw/main/example/reference_4.wav"
)


def make_api_call(inputs):
    data = {"input": inputs}
    resp = requests.post(url, json=data)
    if resp.status_code == 200:
        return resp.json()["output"]
    else:
        raise Exception(resp.json())


def decode_and_save_output(output, filename):
    base64_audio = output.split(",")[1]  # remove the header
    decoded_audio = base64.b64decode(base64_audio)  # decode
    with open(filename, "wb") as f:  # save the audio file
        f.write(decoded_audio)


print("Test 1: Text-to-speech")
params = {
    "input_text": "Hello World!. Now you can synthesize speech from text.",
    "target_voice": reference1,
    "scale_output_volume": True,
    "seed": 1234,
    "output_sr": 48000,
    "denoise_ratio": 0.7,
}

output = make_api_call(params)
decode_and_save_output(output, "output_tts.wav")
print("Saved output to output_tts.wav")

print("Test 2: Voice conversion")
params = {
    "input_sound": reference2,
    "target_voice": reference3,
    "scale_output_volume": False,
    "seed": 541645,
    "output_sr": 16000,
    "denoise_ratio": 0.0,
}
output = make_api_call(params)
decode_and_save_output(output, "output_vc.wav")
print("Saved output to output_vc.wav")
