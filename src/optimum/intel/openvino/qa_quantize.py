import os
import numpy as np
import yaml
import logging as log
import torch

from optimum.intel.openvino.quantize_utils import QADataLoader

from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline
from compression.utils.logger import init_logger
from compression.api import DataLoader
from compression.api import Metric

from addict import Dict

from datasets import load_dataset, load_metric

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class OVQADataLoader(DataLoader):

    # Required methods
    def __init__(self, config):
        """Constructor
        :param config: data loader specific config
        """
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self.task = config['dataset']['task'].lower()
        self.batch_size = config['dataset']['batch_size']
        self.max_length = config['dataset']['max_length']
        self.model_name = config['model_name']
        self.calib_size = config['dataset']['calib_size']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.features = []
        self.labels = []
        self.qa_dataloader = QADataLoader(self.tokenizer, self.max_length)
        self.prepare_dataset()
        self.items = np.arange(self.calib_size)

        print(bcolors.UNDERLINE + "\nQuantizing FP32 OpenVINO model to INT8\n" + bcolors.ENDC)

        print(bcolors.OKBLUE + "There are {:,} samples in the test dataset ".format(len(self.items)) + \
            bcolors.OKGREEN + bcolors.ENDC)

    def __len__(self):
        """Returns size of the dataset"""
        return len(self.items)

    def __getitem__(self, item):
        """
        Returns annotation, data and metadata at the specified index.
        Possible formats:
        (index, annotation), data
        (index, annotation), data, metadata
        """
        if item >= len(self):
            raise IndexError

        input_ids = self.features[self.items[item]]["input_ids"]
        attention_mask = self.features[self.items[item]]["attention_mask"]

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}       
        label = self.labels[self.items[item]]
        return (item, label), inputs

    def prepare_dataset(self):
        """Prepare dataset"""
        # Dataset loading
        self.dataset = load_dataset(self.task, split="validation")

        calibration_features = self.qa_dataloader.get_calib_features(self.dataset)

        for i, feature in enumerate(calibration_features):
            self.features.append({'input_ids': feature["input_ids"], 'attention_mask': feature["attention_mask"]})
            self.labels.append({'input_ids': feature["input_ids"], 'answer': feature["answers"], 'id': feature["id"]})


class OVQAAccuracyMetric(Metric):
    # Required methods
    def __init__(self, config):
        super().__init__()
        self.metric_scores = []
        self.task = config['dataset']['task'].lower()
        self.max_length = config['dataset']['max_length']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.metric = load_metric(self.task)
        self.metric_name = "f1"
        self.qa_dataloader = QADataLoader(self.tokenizer, self.max_length)

    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self.metric_name: self.metric_scores[-1]}

    @property
    def avg_value(self):
        """Returns accuracy metric value for all model outputs."""
        return {self.metric_name: np.ravel(self.metric_scores).mean()}

    def update(self, output, target):
        """
        Updates prediction matches.

        :param output: model output
        :param target: annotations
        """
        predictions, references = self.qa_dataloader.postprocess_function(output, target)
        final_score = self.metric.compute(predictions=predictions, references=references)
        self.metric_scores.append(final_score[self.metric_name])

    def reset(self):
        """
        Resets collected matches
        """
        self.metric_scores = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self.metric_name: {"direction": "higher-better", "type": "accuracy"}}

class OVQuantizerForQuestionAnswering():
    def __init__(self, model_name: str, xml_path, bin_path, config_path: str):
        self.path = config_path
        self.config = self.parse_config()
        self.config['model_name'] = model_name 
        self.model_config = Dict({"model_name": model_name, "model": xml_path, "weights": bin_path})
        self.engine_config = Dict({"device": "CPU"})
        self.algorithms = [
            {
                "name": "AccuracyAwareQuantization", # compression algorithm name
                "params": {
                    "target_device": "CPU",
                    "preset": "performance",
                    "stat_subset_size": 10,
                    "model_type": "transformer",
                    "metric_subset_ratio": 0.5, # A part of the validation set that is used to compare full-precision and quantized models
                    "ranking_subset_size": 300, # A size of a subset which is used to rank layers by their contribution to the accuracy drop
                    "max_iter_num": 10,    # Maximum number of iterations of the algorithm (maximum of layers that may be reverted back to full-precision)
                    "maximal_drop": self.config['quantization']["maximum_metric_drop"],      # Maximum metric drop which has to be achieved after the quantization
                    "drop_type": "absolute",    # Drop type of the accuracy metric: relative or absolute (default)
                    "use_prev_if_drop_increase": True,     # Whether to use NN snapshot from the previous algorithm iteration in case if drop increases
                    "base_algorithm": "DefaultQuantization" # Base algorithm that is used to quantize model at the beginning
                }
            }
        ]


    def parse_config(self):
        with open(self.path) as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                log.error(err)

        return config

    def quantize(self):
        # Step 1: Load the model.
        model = load_model(self.model_config)

        # Step 2: Initialize the data loader.
        data_loader = OVQADataLoader(self.config)
        
        # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
        metric = OVQAAccuracyMetric(self.config)

        # Step 4: Initialize the engine for metric calculation and statistics collection.
        engine = IEEngine(config=self.engine_config,
                          data_loader=data_loader,
                          metric=metric)

        # Step 5: Create a pipeline of compression algorithms.
        pipeline = create_pipeline(self.algorithms, engine)

        metric_results_FP32 = pipeline.evaluate(model)

        # print metric value
        if metric_results_FP32:
            for name, value in metric_results_FP32.items():
                print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)


        # Step 6: Execute the pipeline.
        compressed_model = pipeline.run(model)

        # Step 7 (Optional): Compress model weights to quantized precision
        #                    in order to reduce the size of final .bin file.
        compress_model_weights(compressed_model)

        # Step 8: Save the compressed model to the desired path.
        save_model(compressed_model, os.path.join(os.path.curdir, 'optimized'))

        # Step 9 (Optional): Evaluate the compressed model. Print the results.
        metric_results_INT8 = pipeline.evaluate(compressed_model)

        print(bcolors.BOLD + "\nFINAL RESULTS" + bcolors.ENDC)

        if metric_results_INT8:
            for name, value in metric_results_INT8.items():
                print(bcolors.OKGREEN + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)

        return os.path.join(os.path.curdir, 'optimized')
