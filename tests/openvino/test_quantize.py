import unittest

import numpy as np

import logging as log

import subprocess
import sys
import yaml

from transformers import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST, AutoTokenizer
from transformers.testing_utils import require_tf
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from optimum.intel.openvino import (
    OVAutoModel,
    OVAutoModelForMaskedLM,
    OVAutoModelForQuestionAnswering,
    OVAutoModelWithLMHead,
    OVQuantizerForQuestionAnswering,
)

from datasets import load_dataset, load_metric

from openvino.inference_engine import IECore

ie = IECore()

log.basicConfig(level=log.INFO)

class OVBertForQuestionAnsweringTest(unittest.TestCase):
    def check_model(self, model, tok):
        context = """
        Soon her eye fell on a little glass box that
        was lying under the table: she opened it, and
        found in it a very small cake, on which the
        words “EAT ME” were beautifully marked in
        currants. “Well, I’ll eat it,” said Alice, “ and if
        it makes me grow larger, I can reach the key ;
        and if it makes me grow smaller, I can creep
        under the door; so either way I’ll get into the
        garden, and I don’t care which happens !”
        """

        question = "Where Alice should go?"

        # For better OpenVINO efficiency it's recommended to use fixed input shape.
        # So pad input_ids up to specific max_length.
        input_ids = tok.encode(
            question + " " + tok.sep_token + " " + context, return_tensors="pt", max_length=128, padding="max_length"
        )

        outputs = model(input_ids)

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        answer_ids = input_ids[0, start_pos:end_pos]
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(answer_ids))

        print(answer)

        self.assertEqual(answer, "the garden")


    def test_from_ov_i8(self):
        tok = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-cased-distilled-squad", from_pt=True
        )

        with open('config.yml') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                log.error(err)

        max_length = config['dataset']['max_length']

        subprocess.run(
            [
                sys.executable,
                "-m",
                "mo",
                "--input_model={}".format(model.onnx_path),
                "--model_name=ov_model",
                "--input",
                "input_ids,attention_mask",
                "--input_shape",
                "[1, {}], [1, {}]".format(max_length, max_length),
                "--disable_nhwc_to_nchw",
            ],
            check=True,
        )

        int8_model_dir = OVQuantizerForQuestionAnswering(
            "distilbert-base-cased-distilled-squad", 'ov_model.xml', 'ov_model.bin', 'config.yml'
        ).quantize()

        int8_model = OVAutoModelForQuestionAnswering.from_pretrained(
            '{}'.format(int8_model_dir)
        )

        self.check_model(int8_model, tok)

if __name__ == "__main__":
    unittest.main()