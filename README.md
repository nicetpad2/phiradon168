# phiradon168
ระบบเทรดอัตโนมัติ

โปรเจ็กต์นี้ได้แยกไฟล์ `gold ai 3_5.py` ออกเป็นโมดูลย่อย 5 ไฟล์ในโฟลเดอร์ `src` เพื่อให้ง่ายต่อการปรับปรุงแก้ไข โดยใช้พาธดังนี้:

- `/content/drive/MyDrive/Phiradon168/XAUUSD_M1.csv`
- `/content/drive/MyDrive/Phiradon168/XAUUSD_M15.csv`
- `/content/drive/MyDrive/Phiradon168/logs`

โมดูลที่ได้ประกอบด้วย:
- `config.py` – กำหนดค่าพาธและตั้งค่า logging
- `data_loader.py` – โหลดข้อมูลจากไฟล์ CSV
- `features.py` – สร้างฟีเจอร์เพิ่มเติม
- `strategy.py` – ตรรกะการซื้อขายพื้นฐาน
- `main.py` – จุดเริ่มต้นของโปรแกรม

เรียกใช้โปรแกรมได้ด้วยคำสั่ง:
```bash
python -m src.main
```
