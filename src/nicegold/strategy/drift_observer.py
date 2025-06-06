class DriftObserver:
    """ตัวอย่างคลาสสังเกตการณ์ drift แบบย่อ"""

    def __init__(self, features_to_observe):
        if not isinstance(features_to_observe, list):
            raise ValueError("features_to_observe must be a list")
        self.features = features_to_observe
        self.results = {}
