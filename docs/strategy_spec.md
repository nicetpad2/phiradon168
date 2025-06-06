# Strategy Specification

เอกสารนี้อธิบาย pseudocode สำหรับกฎการเข้าออกออเดอร์ ระบบ OMS และการจัดการความเสี่ยง

## Entry Rule
```text
if Close[i] > Close[i-1] and MACD_hist[i] > 0 and RSI[i] > 50:
    signal = 1
else:
    signal = 0
```

## Exit Rule
```text
if Close[i] < Close[i-1] or MACD_hist[i] < 0 or RSI[i] < 50:
    exit = 1
else:
    exit = 0
```

## OMS & Risk
- ทุกคำสั่งซื้อขายต้องกำหนด SL/TP
- ขนาดล็อตคำนวณจาก `balance * risk_pct / stop_loss_distance`
