# YOLOv8 Training Project

## 📌 Опис
Цей проєкт містить код і структуру файлів для навчання моделі YOLOv8 на власному наборі даних. Використовується бібліотека `ultralytics` для тренування нейромережі, яка зможе розпізнавати об'єкти на зображеннях.

---

## 📂 Структура проєкту
```
project_directory/
│── dataset/
│   ├── images/
│   │   ├── train/  # Тренувальні зображення
│   │   ├── val/    # Валідаційні зображення
│   ├── labels/
│   │   ├── train/  # Анотації для тренувальних зображень
│   │   ├── val/    # Анотації для валідаційних зображень
│── config.yaml      # Файл конфігурації для YOLO
│── train.py         # Скрипт для навчання
│── requirements.txt # Залежності
```

---

## 📋 Вимоги
Перед запуском потрібно встановити необхідні бібліотеки:
```bash
pip install -r requirements.txt
```
Файл `requirements.txt` містить:
```
ultralytics
opencv-python
torch
torchvision
```

---

## 🔧 Налаштування `config.yaml`
Файл `config.yaml` містить інформацію про набір даних:
```yaml
path: ./dataset  
train: images/train  
val: images/val  
nc: 1  # Кількість класів
names: ["drone"]  # Назва класу
```

---

## 🚀 Запуск навчання
Запустити навчання можна командою:
```bash
python train.py
```
Файл `train.py` виконує тренування моделі:
```python
from ultralytics import YOLO

# Завантажуємо YOLOv8 (нано-версія)
model = YOLO("yolov8n.pt")

# Навчання моделі
model.train(data="config.yaml", epochs=50, imgsz=640, batch=16, device="cuda")
```

---

## 📝 Формат анотацій (labels)
Кожне зображення має відповідний `.txt` файл у папці `labels/`.
Формат анотацій:
```
<class> <x_center> <y_center> <width> <height>
```
Приклад (`labels/train/image1.txt` для одного дрона):
```
0 0.5 0.5 0.2 0.3
```
- `0` – клас (перший у `names`)
- `0.5 0.5` – координати центру bounding box
- `0.2 0.3` – ширина та висота (у відносних координатах)

---

## 📊 Результати
Після тренування модель збережеться у `runs/train/exp/weights/best.pt`. Її можна використовувати для інференсу:
```python
model = YOLO("runs/train/exp/weights/best.pt")
results = model.predict("test.jpg", save=True, conf=0.5)
```

---

## ✅ Готово!
Якщо є питання – запитуй! 🔥

