# -*- coding: utf-8 -*-
from modelhub.framework import ApiModel


class Model(ApiModel):
    model_name = "code_practice"

    INPUTS_SAMPLE = None
    OUTPUTS_SAMPLE = None

    def prepare(self):~
        super().prepare()
        # do prepare

    def is_ready(self):
        # check is ready

    def validate_input_data(self, raw_input):
        # do validation
        pass
        return True

    def run_model(self, preprocessed_data):
        # do run
        pass


if __name__=="__main__":
    model = Model()
    output = model.run(model.INPUTS_SAMPLE)
    print(output)