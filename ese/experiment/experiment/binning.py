# local imports
from .utils import load_experiment, parse_class_name
from .posthoc_exp import PostHocExperiment
# torch imports
import torch
# IonPy imports
from ionpy.util import Config
from ionpy.nn.util import num_params
from ionpy.util.ioutil import autosave
from ionpy.analysis import ResultsLoader
from ionpy.experiment.util import eval_config
# misc imports
import os


# Very similar to BaseExperiment, but with a few changes.
class BinningInferenceExp(PostHocExperiment):

    def build_model(self):
        # Move the information about channels to the model config.
        # by popping "in channels" and "out channesl" from the data config and adding them to the model config.
        total_config = self.config.to_dict()
        ###################
        # BUILD THE MODEL #
        ###################
        # Get the configs of the experiment
        load_exp_cfg = {
            "device": "cuda",
            "load_data": False, # Important, we might want to modify the data construction.
        }
        if "config.yml" in os.listdir(total_config['train']['pretrained_dir']):
            self.pretrained_exp = load_experiment(
                path=total_config['train']['pretrained_dir'],
                **load_exp_cfg
            )
        else:
            rs = ResultsLoader()
            dfc = rs.load_configs(
                total_config['train']['pretrained_dir'],
                properties=False,
            )
            self.pretrained_exp = load_experiment(
                df=rs.load_metrics(dfc),
                selection_metric=total_config['train']['pretrained_select_metric'],
                **load_exp_cfg
            )
        pretrained_cfg = self.pretrained_exp.config.to_dict()
        #########################################
        #            Model Creation             #
        #########################################
        model_config_dict = total_config['model']
        if '_pretrained_class' in model_config_dict:
            model_config_dict.pop('_pretrained_class')
        self.model_class = model_config_dict['_class']
        # Either keep training the network, or use a post-hoc calibrator.
        if self.model_class == "Vanilla":
            self.base_model = torch.nn.Identity()
            # Load the model, there is no learned calibrator.
            self.model = self.pretrained_exp.model
            self.properties["num_params"] = num_params(self.model)
        else:
            self.base_model = self.pretrained_exp.model
            self.base_model.eval()
            # Get the old in and out channels from the pretrained model.
            model_config_dict["num_classes"] = pretrained_cfg['model']['out_channels']
            model_config_dict["image_channels"] = pretrained_cfg['model']['in_channels']
            # BUILD THE CALIBRATOR #
            ########################
            # Load the model
            self.model = eval_config(model_config_dict)
            self.model.weights_init()
            self.properties["num_params"] = num_params(self.model) + num_params(self.base_model)
        ########################################################################
        # Make sure we use the old experiment seed and add important metadata. #
        ########################################################################
        old_exp_config = self.pretrained_exp.config.to_dict() 
        total_config['experiment'] = old_exp_config['experiment']
        model_config_dict['_class'] = self.model_class
        model_config_dict['_pretrained_class'] = parse_class_name(str(self.base_model.__class__))
        autosave(total_config, self.path / "config.yml") # Save the new config because we edited it.
        self.config = Config(total_config)