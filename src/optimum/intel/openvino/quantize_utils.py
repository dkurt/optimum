import numpy as np
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class QADataLoader:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_calib_features(self, dataset):
        return dataset.map(
                self.preprocess_function,
                batched=True
            )

    def preprocess_function(self, examples):
        example_questions = [q.strip() for q in examples["question"]]
        example_answers = examples["answers"]
        tokenized_inputs = self.tokenizer(
            example_questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        for i in range(len(tokenized_inputs["input_ids"])):
            seq_ids = tokenized_inputs.sequence_ids(i)
            ctx_start_index = 0
            ctx_end_index = 0
            # Get context ids, but not including padding
            for j, x in enumerate(seq_ids):
                if x == 1:
                    ctx_start_index = j
                    break
            for j, x in enumerate(seq_ids[ctx_start_index:]):
                if x != 1:
                    ctx_end_index = j
                break

            for k, offset in enumerate(tokenized_inputs["offset_mapping"][i]):
                # Only include the offset mapping if it is within the valid context window
                if k > ctx_start_index and k < ctx_end_index:
                    tokenized_inputs["offset_mapping"][i] = offset
                else:
                    tokenized_inputs["offset_mapping"][i] = (0, 0)

        tokenized_inputs["id"] = examples["id"]
        return tokenized_inputs

    def postprocess_function(self, output, target):
        result = QuestionAnsweringModelOutput()
        result["start_logits"] = output["output_s"]
        result["end_logits"] = output["output_e"]
        outputs = result

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        target_id = target[0]['id']
        input_ids = target[0]['input_ids']
        target_answers = target[0]['answer']

        answer_ids = np.array(input_ids)[start_pos:end_pos]
        pred_answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(answer_ids))

        predictions = [{'prediction_text': pred_answer, 'id': target_id}]
        references = [{'answers': target_answers, 'id': target_id}]

        return predictions, references