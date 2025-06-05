import sys
from src.utils import load_json_with_comments


def validate_features(path: str) -> bool:
    """ตรวจสอบความสมบูรณ์ของไฟล์ features_main.json"""
    try:
        features = load_json_with_comments(path)
    except Exception as e:
        print(f"โหลดไฟล์ไม่สำเร็จ: {e}")
        return False
    if not isinstance(features, list):
        print("รูปแบบไม่ถูกต้อง: ต้องเป็น list")
        return False
    if not all(isinstance(f, str) for f in features):
        print("พบรายการที่ไม่ใช่ string")
        return False
    dupes = {f for f in features if features.count(f) > 1}
    if dupes:
        print(f"พบ feature ซ้ำ: {sorted(dupes)}")
        return False
    print("ไฟล์ features_main.json ถูกต้อง")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "features_main.json"
    ok = validate_features(target)
    sys.exit(0 if ok else 1)
