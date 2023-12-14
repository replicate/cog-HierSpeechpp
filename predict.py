import os
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from cog import BasePredictor, Input, Path

import file_utils
from hierspeechpp.inference import HierSpeechppPredictor


WEIGHTS_MAP = {
    "HIER_SPEECH_PP": {
        "path": "weights/hierspeechpp_eng_kor/hierspeechpp_v1.1_ckpt.pth",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/hierspeechpp_eng_kor_v.1.1.tar",
    },
    "DENOISER": {
        "path": "weights/denoiser/g_best",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/denoiser.tar",
    },
    "TEXT2W2V": {
        "path": "weights/ttv_libritts_v1/ttv_lt960_ckpt.pth",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/ttv_libritts_v1.tar",
    },
    "WAV2VEC": {
        "path": "weights/fb-mms-300m/pytorch_model.bin",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/fb-mms-300m.tar",
    },
    "SUPER_RESOLUTION_24K": {
        "path": "weights/speechsr24k/G_340000.pth",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/speechsr24k.tar",
    },
    "SUPER_RESOLUTION_48K": {
        "path": "weights/speechsr48k/G_100000.pth",
        "url": "https://weights.replicate.delivery/default/HierSpeech++/speechsr48k.tar",
    },
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient. Download the weights if they are not already downloaded."""
        for k, v in WEIGHTS_MAP.items():
            if not os.path.exists(v["path"]):
                file_utils.download_and_extract(v["url"], os.path.dirname(v["path"]))

        wav2vec_model_dir = os.path.dirname(WEIGHTS_MAP["WAV2VEC"]["path"])
        self.model = HierSpeechppPredictor(
            ckpt=WEIGHTS_MAP["HIER_SPEECH_PP"]["path"],
            ckpt_text2w2v=WEIGHTS_MAP["TEXT2W2V"]["path"],
            ckpt_wav2vec=wav2vec_model_dir,
            ckpt_sr24=WEIGHTS_MAP["SUPER_RESOLUTION_24K"]["path"],
            ckpt_sr48=WEIGHTS_MAP["SUPER_RESOLUTION_48K"]["path"],
            ckpt_denoiser=WEIGHTS_MAP["DENOISER"]["path"],
        )

    def set_seed(self, seed):
        if seed is None:
            seed = np.random.randint(1, 10000)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    def predict(
        self,
        input_text: str = Input(
            description="Text input to the model. If provided, it will be used for the speech content of the output.",
            default=None,
        ),
        input_sound: Path = Input(
            description="Sound input to the model. If provided, it will be used for the speech content of the output.",
            default=None,
        ),
        target_voice: Path = Input(
            description="A voice clip containing the speaker to synthesize",
        ),
        denoise_ratio: float = Input(
            description="Noise control. 0 means no noise reduction, 1 means maximum noise reduction. \
                If noise reduction is desired, it is recommended to set this value to 0.6~0.8",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        text_to_vector_temperature: float = Input(
            description="Temperature for text-to-vector model. Larger value corresponds to slightly more random output.",
            ge=0.0,
            le=1.0,
            default=0.33,
        ),
        voice_conversion_temperature: float = Input(
            description="Temperature for the voice conversion model. Larger value corresponds to slightly more random output.",
            ge=0.0,
            le=1.0,
            default=0.33,
        ),
        output_sample_rate: int = Input(
            description="Sample rate of the output audio file",
            choices=[16000, 24000, 48000],
            default=16000,
        ),
        scale_output_volume: bool = Input(
            description="Scale normalization. If set to true, the output audio will be scaled according to the input sound if provided.",
            default=False,
        ),
        seed: int = Input(
            description="Random seed to use for reproducibility", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert (
            input_text is not None or input_sound is not None
        ), "Either input_text or input_sound must be provided"

        if input_text is not None:
            assert len(input_text) < 200, "Input text must be less than 200 characters"

        self.set_seed(seed)  # set seed for reproducibility

        # scale norm: "prompt" if output is to be scaled according to the input sound,
        # "max" if output is to be scaled according to the maximum value of the output
        scale_norm = "prompt" if scale_output_volume else "max"

        # if input sound is not provided, run text-to-speech
        if input_sound is None:
            output = self.model.text_to_speech(
                text=input_text,
                target_voice=str(target_voice),
                denoise_ratio=denoise_ratio,
                noise_scale_ttv=text_to_vector_temperature,
                noise_scale_vc=voice_conversion_temperature,
                scale_norm=scale_norm,
                output_sr=output_sample_rate,
            )
        # if input sound is provided, run voice conversion
        else:
            output = self.model.voice_conversion(
                source_speech=str(input_sound),
                target_voice=str(target_voice),
                denoise_ratio=denoise_ratio,
                noise_scale_vc=voice_conversion_temperature,
                scale_norm=scale_norm,
                output_sr=output_sample_rate,
            )

        output_path = "/tmp/output.wav"
        write_wav(output_path, output_sample_rate, output)

        return Path(output_path)
