from dataclasses import dataclass, field
import os
import yaml

# [Patch v5.5.14] Simple dataclass-based pipeline config loader

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, 'pipeline.yaml')


@dataclass
class DataConfig:
    """Data related configuration."""
    output_dir: str = 'output_default'
    features_filename: str = 'features_main.json'
    trade_log_pattern: str = 'trade_log_*.csv*'


@dataclass
class PipelineConfig:
    log_level: str = 'INFO'
    model_dir: str = 'models'
    threshold_file: str = 'threshold_wfv_optuna_results.csv'
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str = DEFAULT_CONFIG_FILE) -> 'PipelineConfig':
    """Load configuration from YAML file if available."""
    cfg = PipelineConfig()
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        if 'log_level' in data:
            cfg.log_level = data['log_level']
        if 'model_dir' in data:
            cfg.model_dir = data['model_dir']
        if 'threshold_file' in data:
            cfg.threshold_file = data['threshold_file']
        if 'data' in data:
            d = data['data'] or {}
            cfg.data.output_dir = d.get('output_dir', cfg.data.output_dir)
            cfg.data.features_filename = d.get('features_filename', cfg.data.features_filename)
            cfg.data.trade_log_pattern = d.get('trade_log_pattern', cfg.data.trade_log_pattern)
    return cfg
