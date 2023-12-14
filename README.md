# Cog wrapper for HierSpeech++

A Cog wrapper for HierSpeech++, a text-to-speech model that can generate speech from text and a target voice for zero-shot speech synthesis. See the original [repository](https://arxiv.org/abs/2311.12454), [paper](https://arxiv.org/abs/2311.12454) and [Replicate demo](https://replicate.com/adirik/hierspeechpp) for details.

## API Usage
You need to have Cog and Docker installed to run this model locally. Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of HierSpeech++ to [Replicate](https://replicate.com).
To use the model, simply provide the text you would like to generate speech and a sound file of your target voice as input. Optionally provide a reference speech (.mp3 or .wav) instead of text to parse speech content. The API returns an .mp3 file with generated speech.

To build the docker image with cog and run a prediction:
```bash
cog predict -i input_text="This is a zero-shot text to speech model." -i target_voice=@examples/reference_1.wav
```

To start a server and send requests to your locally or remotely deployed API:
```bash
cog run -p 5000 python -m cog.server.http
```

Input parameters are as follows:  
- **input_text:** (optional) text input to the model. If provided, it will be used for the speech content of the output.
- **input_sound:** (optional) sound input to the model. If provided, it will be used for the speech content of the output..  
- **target_voice:** a voice clip containing the speaker to synthesize.
- **denoise_ratio:** noise control. 0 means no noise reduction, 1 means maximum noise reduction. If noise reduction is desired, it is recommended to set this value to 0.6~0.8.  
- **text_to_vector_temperature:** temperature for text-to-vector model. Larger value corresponds to slightly more random output.
- **output_sample_rate:** sample rate of the output audio file.
- **scale_output_volume:** scale normalization. If set to true, the output audio will be scaled according to the input sound if provided.
- **seed:** random seed to use for reproducibility.


## References 
```
@article{Lee2023HierSpeechBT,
  title={HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis},
  author={Sang-Hoon Lee and Haram Choi and Seung-Bin Kim and Seong-Whan Lee},
  journal={ArXiv},
  year={2023},
  volume={abs/2311.12454},
  url={https://api.semanticscholar.org/CorpusID:265308903}
}
```
