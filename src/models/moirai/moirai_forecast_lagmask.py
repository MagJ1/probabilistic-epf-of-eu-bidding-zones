
from uni2ts.model.moirai.forecast import MoiraiForecast
from gluonts.transform import Chain  # if not already imported
from models.moirai.custom_transforms import MaskContextTailPostSplit
from gluonts.torch import PyTorchPredictor
from gluonts.transform.split import TFTInstanceSplitter

from gluonts.transform import TestSplitSampler

class MoiraiForecastLagMask(MoiraiForecast):

    def __init__(
            self,
            *args,
            lag_mask_steps = 14,
            lag_mask_value = 0.0,
            **kwargs
            
    ):
        super().__init__(*args,**kwargs)
        self._lag_mask_steps = lag_mask_steps
        self._lag_mask_value = lag_mask_value


    def create_predictor(
        self,
        batch_size: int,
        device: str = "auto",
    ) -> PyTorchPredictor:
        ts_fields = []
        if self.hparams.feat_dynamic_real_dim > 0:
            ts_fields.append("feat_dynamic_real")
            ts_fields.append("observed_feat_dynamic_real")
        past_ts_fields = []
        if self.hparams.past_feat_dynamic_real_dim > 0:
            past_ts_fields.append("past_feat_dynamic_real")
            past_ts_fields.append("past_observed_feat_dynamic_real")

        instance_splitter = TFTInstanceSplitter(
            instance_sampler=TestSplitSampler(),
            past_length=self.past_length,
            future_length=self.hparams.prediction_length,
            observed_value_field="observed_target",
            time_series_fields=ts_fields,
            past_time_series_fields=past_ts_fields,
        )

        # Base transform (same as before)
        input_tf = self.get_default_transform() + instance_splitter

        # NEW
        if self._lag_mask_steps and self.hparams.past_feat_dynamic_real_dim > 0:
            input_tf = input_tf + MaskContextTailPostSplit(
                steps=int(self._lag_mask_steps),
                context_length=int(self.hparams.context_length),
                fill_value=float(self._lag_mask_value),
            )

        return PyTorchPredictor(
            input_names=self.prediction_input_names,
            prediction_net=self,
            batch_size=batch_size,
            prediction_length=self.hparams.prediction_length,
            input_transform=input_tf,
            device=device,
        )