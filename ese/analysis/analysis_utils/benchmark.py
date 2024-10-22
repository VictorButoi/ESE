

def load_benchmark_params(base_cfg, determiner):

        inference_type, parameter = determiner.split("_")

        if inference_type.lower() == "sweep":
            if parameter.lower() == "threshold":
                exp_cfg_update = {
                    "experiment": {
                        "inf_kwargs": {
                            "threshold": [ 
                                (0.00, ..., 0.25, 0.025),
                                (0.26, ..., 0.50, 0.025),
                                (0.51, ..., 0.75, 0.025),
                                (0.76, ..., 1.00, 0.025)
                            ]
                        }
                    }
                }
            elif parameter.lower() == "temperature":
                exp_cfg_update = {
                    "experiment": {
                        "inf_kwargs": {
                            "temperature": [ 
                                (0.01, ..., 0.50, 0.025),
                                (0.51, ..., 1.00, 0.025), 
                                (1.01, ..., 1.25, 0.025),
                                (1.26, ..., 1.50, 0.025),
                                (1.51, ..., 2.00, 0.025),
                                (2.01, ..., 2.50, 0.025),
                                (2.51, ..., 2.75, 0.025),
                                (2.76, ..., 3.00, 0.025),
                            ]
                        }
                    }
                }
            else:
                raise ValueError(f"Unknown parameter for sweep inference: {parameter}.")
        elif inference_type.lower() == "optimal":
            if parameter.lower() == "threshold":
                raise NotImplementedError("Optimal threshold inference not implemented.")
                # exp_cfg_update = {
                #     "experiment": {
                #         "inf_kwargs": {
                #             "threshold": get_optimal_parameter()
                #         }
                #     }
                # }
            elif parameter.lower() == "temperature":
                raise NotImplementedError("Optimal threshold inference not implemented.")
                # exp_cfg_update = {
                #     "experiment": {
                #         "inf_kwargs": {
                #             "threshold": get_optimal_parameter()
                #         }
                #     }
                # }
            else:
                raise ValueError(f"Unknown parameter for optimal inference: {parameter}.")
        else:
            raise ValueError(f"Unknown inference type: {inference_type}.")
        
        # Update the experiment config.
        base_cfg.update(exp_cfg_update)

        return base_cfg