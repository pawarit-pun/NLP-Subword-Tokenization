# NLP Subword Tokenization

เปรียบเทียบ BPE, WordPiece และ Unigram โดย Train บนข้อมูลชุดเดียวกัน

## Data

- **Dataset:** WikiText-103 (`Salesforce/wikitext`, config `wikitext-103-raw-v1`) — บทความ Wikipedia ภาษาอังกฤษ
- **Split:** ใช้ Split ทางการของ Dataset ไม่สุ่มแบ่งใหม่

| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `data/train.txt` | Train Tokenizer ทั้ง 3 แบบ |
| `data/validation.txt` | ลองค่า / เลือก Config |
| `data/test.txt` | ประเมินผลตอนท้ายเท่านั้น ห้ามใช้ Train |

รูปแบบ: UTF-8, 1 บรรทัด = 1 ข้อความ

## วิธีสร้างไฟล์

```bash
pip install -r requirements.txt
python preprocess/preprocess.py
```

## Preprocessing ทำอะไร

1. ลบบรรทัดว่าง
2. หัวข้อ `= = Gameplay = =` → `Gameplay`
3. Unicode NFC
4. แปลงเครื่องหมายพิเศษของ WikiText กลับ: `1 @,@ 000` → `1,000`, `well @-@ known` → `well-known`, `3 @.@ 5` → `3.5`
5. ลบช่องว่างหน้าเครื่องหมาย: `game 's story ,` → `game's story,`
6. ลบบรรทัดใน train ที่ซ้ำกับ validation/test (กันข้อมูลรั่ว)

**ไม่ทำ:** Lowercase, ลบเครื่องหมาย, ลบตัวเลข, ลบ Stopwords เพราะ Tokenizer ต้องเรียนรู้สิ่งเหล่านี้
