# Conception Bank — Stockage des Mask Slow et Fast

## 1 Extension du contrat `Mask`

### 1 Contexte : `Mask` dans `mask.py`

**Fichier** : `core/mask.py`,

## 2 Extension du contrat `FastMaskView`

### 1 Contexte : `to_fast_view()` dans `mask.py`

**Fichier** : `core/mask.py`, fonction `Mask.to_fast_view()`

## 3 Schéma JSON complet + stratégie flush non-bloquant

### 1 Banque = fichier JSONL (un objet JSON par ligne)

Un fichier **JSON Lines** (pas un seul gros JSON) :

- Chaque ligne = un `BankSnapshot` à un instant T
- Pas de verrou partagé entre threads
- Pas de relecture complète à chaque flush

### 2 Stratégie de flush — synchronisée frame (aligné `frame_dumper`)

```txt
┌──────────────────── THREAD MAIN (main.py) ────────────────────────────┐
│  L121  frame_id = capturer.get_frame_id()                             │
│  L123  if frame_id > last_frame_id:            ← garde unique         │
│                                                                       │
│          ── branche slow (L140, si slow_updated) ──                   │
│          bank.add_observation(dets/masks, frame_id, detected_frame_ts)│
│                                                                       │
│          ── branche fast (L153, si not slow_updated) ──               │
│          bank.add_observation(fast_results, frame_id, fast_ts)        │
│                                                                       │
│  L195    frame_dumper.on_events(...)                                  │
│  L196    bench.push_frame()                                           │
│  L197+   bank.flush_if_needed(frame_id)      ← même point             │
│                                                                       │
│  L232  finally: bank.stop()   (flush terminal, avant capturer)        │
└───────────────────────────────────────────────────────────────────────┘
```

**Justification :** aucun thread ni lock supplémentaire ; le push et le flush vivent dans la même garde `if frame_id > last_frame_id` que `frame_dumper` et `bench`, donc **un push exactement par frame produite** — pas de double comptage ni de trou. La cadence de flush est exprimée en **frames** (`bank.flush_every_n_frames`), comparée à `frame_id`, jamais à un compteur d'appels interne. `stop()` dans le `finally` (L232), aux côtés de `frame_dumper.stop()` (L234).

---

## 4 Structure `Bank` (sans template image)

**Absence de `template`** : inchangé, le champ ndarray reste délibérément exclu. Recoupement via `frame_id` → `capturer.get_ring_frame(frame_id)`.

---

## 5 Delta minimal & points d'insertion précis

### 5-A Fichiers à créer / modifier

| Fichier                               | Action     | Raison                                                                                      |
| ------------------------------------- | ---------- | ------------------------------------------------------------------------------------------- |
| `core/bank.py`                        | **CREATE** | Module Bank isolé                                                                           |
| `core/mask.py` — `FastMaskView`       | **EXTEND** | Verfier les champs gratuits manquants                                                       |
| `core/mask.py` — `Mask`               | **EXTEND** | Verfier les champs                                                                          |
| `core/mask.py` — `to_fast_view()`     | **EXTEND** | Verfier les champs                                                                          |
| `core/mask.py` — `serialize_scores()` | **REUSE**  | Existe déjà — sérialise scores pour JSON                                                    |
| `config.yaml` — section `bank:`       | **ADD**    | Flags de config + output path                                                               |
| `main.py`                             | **INSERT** | 4 points : instanciation ; push slow (L140) ; push fast (L155) ; flush + stop (L196 / L233) |

---
