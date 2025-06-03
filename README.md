# Phiradon168

คำแนะนำการติดตั้ง
-------------------
1. สร้าง virtualenv และติดตั้งไลบรารีหลัก:
   ```bash
   pip install -r requirements.txt
   ```
2. หากต้องการให้โปรแกรมติดตั้งไลบรารีอัตโนมัติเมื่อไม่พบ ให้ตั้งค่า `AUTO_INSTALL_LIBS=True` ใน `src/config.py`.
   ค่าเริ่มต้นคือ `False` เพื่อป้องกันการติดตั้งบนระบบที่ไม่มีสิทธิ์หรือไม่มีอินเทอร์เน็ต

## การวัดประสิทธิภาพ
ใช้ `profile_backtest.py` เพื่อวัด bottleneck ของฟังก์ชันจำลองการเทรด
ตัวอย่างการรัน:
```bash
python profile_backtest.py XAUUSD_M1.csv --rows 10000
```
คำสั่งด้านบนจะแสดง 20 ฟังก์ชันที่ใช้เวลามากที่สุดตามค่า `cumtime` จาก `cProfile`.
