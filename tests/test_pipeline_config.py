from src.utils import pipeline_config


def test_load_config_defaults(tmp_path):
    cfg = pipeline_config.load_config(str(tmp_path / 'missing.yaml'))
    assert isinstance(cfg, pipeline_config.PipelineConfig)
    assert cfg.model_dir == 'models'


def test_load_config_override(tmp_path):
    conf_path = tmp_path / 'cfg.yaml'
    conf_path.write_text('log_level: DEBUG\nmodel_dir: demo\nthreshold_file: t.csv')
    cfg = pipeline_config.load_config(str(conf_path))
    assert cfg.log_level == 'DEBUG'
    assert cfg.model_dir == 'demo'
    assert cfg.threshold_file == 't.csv'
