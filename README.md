# Phiradon168

คำแนะนำการติดตั้ง
-------------------
1. สร้าง virtualenv และติดตั้งไลบรารีหลัก:
   ```bash
   pip install -r requirements.txt
   ```
2. หากต้องการให้โปรแกรมติดตั้งไลบรารีอัตโนมัติเมื่อไม่พบ ให้ตั้งค่า `AUTO_INSTALL_LIBS=True` ใน `src/config.py`.
    ค่าเริ่มต้นคือ `True` ช่วยให้ระบบติดตั้งไลบรารีที่ขาดหายได้อัตโนมัติ แต่หากต้องการปิดการติดตั้งเองให้เปลี่ยนเป็น `False`
    และหาก `AUTO_INSTALL_LIBS=False` แล้วไม่พบไลบรารีสำคัญ (CatBoost, Optuna, SHAP)

### การปรับค่า Drift Threshold
หากต้องการปรับเกณฑ์การแจ้งเตือน Drift สำหรับฟีเจอร์ เช่น ADX
สามารถกำหนดตัวแปรสภาพแวดล้อม `DRIFT_WASSERSTEIN_THRESHOLD` (ค่าปกติ `0.1`)
ก่อนรันโปรแกรมได้ เช่น
```bash
export DRIFT_WASSERSTEIN_THRESHOLD=0.2
python ProjectP.py
```


## การวัดประสิทธิภาพ
ใช้ `profile_backtest.py` เพื่อวัด bottleneck ของฟังก์ชันจำลองการเทรด
ตัวอย่างการรัน:
```bash
python profile_backtest.py XAUUSD_M1.csv --rows 10000 --limit 30 --output profile.txt --output-file backtest.prof
```
คำสั่งด้านบนจะแสดง 30 ฟังก์ชันที่ใช้เวลามากที่สุดตามค่า `cumtime` จาก `cProfile` และบันทึกผลไว้ใน `profile.txt` รวมทั้งไฟล์ `backtest.prof` สำหรับเปิดใน SnakeViz.
นอกจากนี้ยังสามารถระบุชื่อ Fund Profile และสั่งให้ฝึกโมเดลหลังจบการทดสอบได้ดังนี้:
```bash
python profile_backtest.py XAUUSD_M1.csv --fund AGGRESSIVE --train --train-output models
```

## การลดข้อความ Log
หากต้องการให้โปรแกรมแสดงเฉพาะคำเตือนและสรุปผลแบบย่อ สามารถตั้งค่า
ตัวแปรสภาพแวดล้อม `COMPACT_LOG=1` ก่อนรัน `ProjectP.py` เช่น

```bash
COMPACT_LOG=1 python ProjectP.py
```
ค่าดังกล่าวจะปรับระดับ log เป็น `WARNING` อัตโนมัติ ทำให้เห็นเฉพาะ
ข้อความสำคัญและผลลัพธ์สรุปท้ายรัน

### การรันชุดทดสอบ
ใช้สคริปต์ `run_tests.py` เพื่อรัน `pytest` โดยเปิดโหมด COMPACT_LOG อัตโนมัติ

```bash
python run_tests.py
```
ผลลัพธ์จะแสดงเฉพาะคำเตือนและสรุปจำนวนการทดสอบทั้งหมด

## การปรับค่า Drift Override
สามารถกำหนดระดับ Drift ที่จะปิดการใช้คะแนน RSI ได้ที่ตัวแปร
`RSI_DRIFT_OVERRIDE_THRESHOLD` ในไฟล์ `src/config.py` (ค่าเริ่มต้น 0.65)
หาก Drift ของ RSI สูงเกินค่านี้ ระบบจะไม่ใช้เงื่อนไข RSI ในการคำนวณสัญญาณ

## หมายเหตุการใช้งาน
* ฟังก์ชัน `safe_set_datetime` ภายใน `data_loader.py` ช่วยแก้ปัญหา
  `FutureWarning` เมื่อต้องตั้งค่า datetime ใน DataFrame
* ฟังก์ชัน `setup_fonts` ถูกออกแบบมาสำหรับ Google Colab เพื่อให้กราฟแสดงฟอนต์ไทยได้ถูกต้อง
  หากใช้งานบน VPS อาจไม่จำเป็น และบางครั้งอาจทำให้เกิด error จากการติดตั้งฟอนต์
* ไฟล์ QA summary (`qa_summary_<label>.log`) และไฟล์แจ้งเตือนกรณีไม่มีข้อมูล (`<label>_trade_qa.log`)
  จะถูกเก็บไว้ภายใน `output_default/qa_logs/` โดยสามารถสร้างโฟลเดอร์ย่อยตามชื่อกองทุน
  เช่น `output_default/qa_logs/FUND_A/` เพื่อแยกข้อมูล QA ของแต่ละกองทุนอย่างเป็นระเบียบ
* หากต้องประมวลผลอินดิเคเตอร์หลายชุดบนข้อมูลขนาดใหญ่ ควรใช้
  `load_data_cached()` เพื่อบันทึกผลลัพธ์ในรูปแบบ Parquet/Feather
  และสามารถบันทึก DataFrame ที่สร้างฟีเจอร์แล้วเป็นไฟล์ HDF5 ผ่าน
  `save_features_hdf5()` เพื่อให้โหลดซ้ำได้เร็วขึ้น

## ภาพรวมกระบวนการทำงาน
เพื่อให้เห็นขั้นตอนหลักของระบบได้ชัดเจนยิ่งขึ้น สามารถอ้างอิงแผนภาพ
Mermaid ด้านล่างซึ่งสรุปการไหลของข้อมูลตั้งแต่การรับข้อมูลดิบไปจนถึงการส่งออกโมเดล

```mermaid
flowchart LR
    A[Data Ingestion] --> B[Feature Engineering]
    B --> C[Backtest]
    C --> D[Meta-Model Training]
    D --> E[Artifact Export]
```

แผนภาพข้างต้นช่วยให้ทีมมองเห็นภาพรวมของกระบวนการได้รวดเร็ว ไม่ว่าจะเป็นการเตรียมข้อมูล
การสร้างฟีเจอร์ การทดสอบย้อนกลับ ไปจนถึงการฝึกเมตาโมเดลและการนำผลลัพธ์ไปใช้งาน

### การวิเคราะห์ Trade Logs
เครื่องมือ `src/log_analysis.py` ช่วยสรุปผลการเทรดจากไฟล์ `logs` ไม่ว่าจะเป็นช่วงเวลาที่ได้กำไรมากที่สุด อัตราการชนะต่อชั่วโมง สาเหตุการปิดออเดอร์ ระยะเวลาถือครอง และสถิติ drawdown
ตัวอย่างการใช้งาน:
```python
from src.log_analysis import (
    parse_trade_logs,
    calculate_hourly_summary,
    calculate_reason_summary,
    calculate_duration_stats,
    calculate_drawdown_stats,
    parse_alerts,
    calculate_alert_summary,
    export_summary_to_csv,
    plot_summary,
)

logs_df = parse_trade_logs('logs')
summary = calculate_hourly_summary(logs_df)
print(summary)
reason_stats = calculate_reason_summary(logs_df)
duration = calculate_duration_stats(logs_df)
drawdown = calculate_drawdown_stats(logs_df)
alerts = calculate_alert_summary('logs')
export_summary_to_csv(summary.reset_index(), 'summary.csv.gz')
fig = plot_summary(summary)
fig.savefig('summary.png')
```
ฟังก์ชัน `calculate_position_size` ยังช่วยคำนวณขนาดลอตที่เหมาะสมตามทุนและระยะ SL

