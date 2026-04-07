import os

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, PowerTransformer

from tfmplayground.model import NanoTabPFNModel, NanoTabPFNModelV1
from tfmplayground.utils import get_default_device


def _convert_inproj_to_separate(sd: dict) -> dict:
    """
    Convert nn.MultiheadAttention in_proj_weight/bias (3e×e) to separate
    q_proj / k_proj / v_proj so the state dict loads into NanoTabPFNModelV1.
    """
    new_sd = {}
    for k, v in sd.items():
        converted = False
        for attn in ("self_attention_between_features", "self_attention_between_datapoints"):
            if k.endswith(f".{attn}.in_proj_weight"):
                prefix = k[:-len(f".{attn}.in_proj_weight")]
                e = v.shape[0] // 3
                new_sd[f"{prefix}.{attn}.q_proj.weight"] = v[:e].clone()
                new_sd[f"{prefix}.{attn}.k_proj.weight"] = v[e:2*e].clone()
                new_sd[f"{prefix}.{attn}.v_proj.weight"] = v[2*e:].clone()
                converted = True
                break
            if k.endswith(f".{attn}.in_proj_bias"):
                prefix = k[:-len(f".{attn}.in_proj_bias")]
                e = v.shape[0] // 3
                new_sd[f"{prefix}.{attn}.q_proj.bias"] = v[:e].clone()
                new_sd[f"{prefix}.{attn}.k_proj.bias"] = v[e:2*e].clone()
                new_sd[f"{prefix}.{attn}.v_proj.bias"] = v[2*e:].clone()
                converted = True
                break
        if not converted:
            new_sd[k] = v
    return new_sd


def init_model_from_state_dict_file(file_path):
    ckpt = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
    arch = ckpt['architecture']
    model_sd = ckpt['model']

    # None buffers are excluded from PyTorch state_dict — extract borders separately
    borders = model_sd.pop('borders', None)

    is_legacy = any(
        'self_attention_between_features' in k or 'self_attention_between_datapoints' in k
        for k in model_sd
    )

    if is_legacy:
        # official checkpoints use nn.MultiheadAttention in_proj_weight; split into q/k/v
        if any('in_proj_weight' in k for k in model_sd):
            model_sd = _convert_inproj_to_separate(model_sd)
        model = NanoTabPFNModelV1(**arch)
    else:
        model = NanoTabPFNModel(
            num_attention_heads=arch['num_attention_heads'],
            embedding_size=arch['embedding_size'],
            mlp_hidden_size=arch['mlp_hidden_size'],
            num_layers=arch['num_layers'],
            num_outputs=arch['num_outputs'],
            residual_decay=arch.get('residual_decay', 1.0),
            num_thinking_rows=arch.get('num_thinking_rows', 0),
            use_qassmax=arch.get('use_qassmax', False),
            use_quantile_loss=arch.get('use_quantile_loss', False),
        )

    model.load_state_dict(model_sd, strict=True)

    # fall back to sibling *-buckets.pth if borders not baked into checkpoint
    if borders is None:
        buckets_path = file_path.replace('-checkpoint.pth', '-buckets.pth')
        if os.path.isfile(buckets_path):
            borders = torch.load(buckets_path, map_location='cpu', weights_only=False)

    if borders is not None:
        model.borders = borders

    return model


def to_pandas(x):
    return pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x


def to_numeric(x):
    return x.apply(pd.to_numeric, errors='coerce').to_numpy()


def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum()
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(to_pandas)),
        ("to_numeric", FunctionTransformer(to_numeric)),
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True))
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor


class NanoTabPFNClassifier():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel | str | None = None, device: None | str | torch.device = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            model = 'checkpoints/nanotabpfn.pth'
            if not os.path.isfile(model):
                os.makedirs("checkpoints", exist_ok=True)
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_classifier.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)
        self.model = model.to(device)
        self.device = device
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        y = self.y_train
        device_type = torch.device(self.device).type if isinstance(self.device, str) else self.device.type
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            out = out[:, :self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to('cpu').float().numpy()


class NanoTabPFNRegressor():
    """ scikit-learn like interface """
    def __init__(self, model: NanoTabPFNModel | str | None = None,
                 dist: FullSupportBarDistribution | str | None = None,
                 device: str | torch.device | None = None, num_mem_chunks: int = 8):
        if device is None:
            device = get_default_device()
        if model is None:
            os.makedirs("checkpoints", exist_ok=True)
            model = 'checkpoints/nanotabpfn_regressor.pth'
            dist = 'checkpoints/nanotabpfn_regressor_buckets.pth'
            if not os.path.isfile(model):
                print('No cached model found, downloading model checkpoint.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor.pth')
                with open(model, 'wb') as f:
                    f.write(response.content)
            if not os.path.isfile(dist):
                print('No cached bucket edges found, downloading bucket edges.')
                response = requests.get('https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/nanotabpfn_regressor_buckets.pth')
                with open(dist, 'wb') as f:
                    f.write(response.content)
        if isinstance(model, str):
            model = init_model_from_state_dict_file(model)

        if isinstance(dist, str):
            bucket_edges = torch.load(dist, map_location=device)
            dist = FullSupportBarDistribution(bucket_edges).float()

        # quantile-loss models have no borders — detect via model flag
        self.use_quantile = getattr(model, 'use_quantile_loss', False)

        if not self.use_quantile:
            if dist is None:
                if model.borders is not None:
                    dist = FullSupportBarDistribution(model.borders.float())
                else:
                    raise ValueError("No dist provided and model has no borders buffer. Pass dist explicitly.")

        self.model = model.to(device)
        self.device = device
        self.dist = dist
        self.num_mem_chunks = num_mem_chunks

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.feature_preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.feature_preprocessor.fit_transform(X_train)
        self.y_train = y_train

        # Yeo-Johnson transform on target
        self.y_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        self.y_train_n = self.y_transformer.fit_transform(y_train.reshape(-1, 1)).ravel()

        # keep z-score stats for the no-transform ensemble member
        self.y_train_mean = np.mean(self.y_train)
        self.y_train_std = np.std(self.y_train, ddof=1) + 1e-8
        self.y_train_z = (self.y_train - self.y_train_mean) / self.y_train_std

    def _forward(self, X_test: np.ndarray, y_context: np.ndarray) -> np.ndarray:
        X = np.concatenate((self.X_train, self.feature_preprocessor.transform(X_test)))
        device_type = torch.device(self.device).type if isinstance(self.device, str) else self.device.type
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
            y_tensor = torch.tensor(y_context, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model((X_tensor, y_tensor), single_eval_pos=len(self.X_train), num_mem_chunks=self.num_mem_chunks).squeeze(0)
            if self.use_quantile:
                # point estimate = mean of 999 quantile predictions
                return logits.float().mean(dim=-1).cpu().float().numpy()
            return self.dist.mean(logits.float()).cpu().float().numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # member 1: Yeo-Johnson transformed target
        preds_yj_n = self._forward(X_test, self.y_train_n)
        # clip to trained range to prevent inverse_transform overflow on untrained models
        yj_min, yj_max = self.y_train_n.min(), self.y_train_n.max()
        preds_yj_n = np.clip(preds_yj_n, yj_min - 3 * abs(yj_min), yj_max + 3 * abs(yj_max))
        preds_yj = self.y_transformer.inverse_transform(preds_yj_n.reshape(-1, 1)).ravel()

        # member 2: plain z-score (original behaviour)
        preds_z_n = self._forward(X_test, self.y_train_z)
        preds_z = preds_z_n * self.y_train_std + self.y_train_mean

        # average ensemble; replace any residual NaN with z-score member
        ensemble = (preds_yj + preds_z) / 2.0
        nan_mask = ~np.isfinite(ensemble)
        ensemble[nan_mask] = preds_z[nan_mask]
        return ensemble
