# Transformer и Vision Transformer

Разбор encoder'а трансформера (*Attention Is All You Need*) и его применение к изображениям (*An Image is Worth 16×16 Words*, ViT)

## Отчёты

| # | Задание | Отчёт |
|---|---|---|
| 1 | Self-attention и encoder-блок Transformer: Q, K, V, scaled dot-product attention, multi-head attention, MLP, residual + LayerNorm, размеры тензоров и стоимость по числу токенов | [report/transformer_encoder.md](report/transformer_encoder.md) |
| 2 | Vision Transformer: конспект статьи, patch embedding, учебный ViT, предобученный ViT-B/16, число токенов при разных разрешениях и патчах | [report/vit.md](report/vit.md) |

## Структура

```
Transformer/
├── transformer_encoder.py   # задание 1: attention, MultiHeadAttention, EncoderBlock (post-LN)
├── vit_patch_embedding.py   # задание 2: нарезка изображения на патчи и проекция в embeddings
├── vit_model.py             # задание 2: учебный ViT на блоке из задания 1 (pre-LN)
├── vit_pretrained.py        # задание 2: предобученный torchvision ViT-B/16, разбор модулей
├── vit_token_count.py       # задание 2: число токенов для разных разрешений и размеров патча
├── report/                  # отчёты по заданиям
└── result_ViT/              # входное изображение и полученные графики
```

## Запуск

Зависимости: `torch`, `torchvision`, `matplotlib`, `pillow`.

Все скрипты запускаются из корня проекта и печатают формы тензоров и результаты проверок:

```bash
python transformer_encoder.py
python vit_patch_embedding.py
python vit_model.py
python vit_pretrained.py      # при первом запуске скачивает веса ViT-B/16 (~330 МБ)
python vit_token_count.py
```

Входное изображение — `result_ViT/img.jpg`; графики сохраняются в ту же папку.
